'''
Apply post-hoc transforms to pre-computed embeddings and evaluate retrieval performance.
'''
import torch
import json
import random
from scipy import stats
from transformers import (
    set_seed,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers.trainer_utils import set_seed
import wandb
import argparse
from utils.load_data import load_beir_data
from utils.logging import LoggingHandler

from utils.beir_custom_evaluation import EvaluateRetrieval
 
from utils.beir_exact_search import DenseRetrievalExactSearch as DRES_GPU
from utils.beir_exact_search import DenseRetrievalExactSearch as DRES_CPU
from utils.beir_utils import DenseEncoderModel, SentenceEncoderModel,SentenceEncoderModel_Prompt
from utils.utils import replace_slash, get_last_element, merge_beir_eval_scores, to_numpy,bright_scores_remove_excluded_ids
import logging
import os
import json
import torch
import sys
import transformers
from torch import nn
from transformers import set_seed
import glob
from beir.retrieval.search.dense.faiss_index import FaissFlatSearcher
from beir.retrieval.search.dense.util import cos_sim, dot_score, pickle_load, save_embeddings
from itertools import chain
from utils.utils import calculate_retrieval_metrics,remove_identical_ids


from embedding_transforms.baselines_numpy import matryoshka_truncation,random_truncation,pca_truncation,whitening_k_truncation,random_projection_truncation
from embedding_transforms.spectemp import unified_spectral_tempering_truncation

import torch
import numpy as np
from tqdm import tqdm
import faiss
logger = logging.getLogger(__name__)
#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

def _extract_shard_index(filepath):
    """Extract numeric shard index from filenames like 'corpus.3.pkl'.
    Fallbacks to 0 if pattern isn't found.
    """
    name = os.path.basename(filepath)
    try:
        parts = name.split('.')
        # Expecting ["corpus", "<idx>", "pkl"]
        return int(parts[1]) if len(parts) > 2 else 0
    except Exception:
        return 0

def load_local_embeddings(corpus_embeddings_file_path=None, query_embeddings_file_path=None):
    # Initialise return variables to None
    corpus_embeddings = None
    all_corpus_ids = None
    query_embeddings = None
    query_ids = None

    if corpus_embeddings_file_path is not None:
        # Load corpus embeddings
        corpus_embeddings_files = glob.glob(f"{corpus_embeddings_file_path}/corpus.*.pkl")
        
        if not corpus_embeddings_files:
            logger.warning(f"No corpus embedding files found in {corpus_embeddings_file_path}")
        else:
            logger.info("Loading precomputed corpus embeddings...")
            # Stable, deterministic ordering (sorted by shard index)
            corpus_embeddings_files.sort(key=_extract_shard_index)

            iterator = (pickle_load(p) for p in corpus_embeddings_files)
            if len(corpus_embeddings_files) > 1:
                iterator = tqdm(iterator, desc="Loading shards into index", total=len(corpus_embeddings_files))

            all_corpus_ids = []
            corpus_embeddings_all_list = []
            for emb, ids in iterator:
                corpus_embeddings_all_list.append(emb)
                all_corpus_ids.extend(ids)
            corpus_embeddings = np.concatenate(corpus_embeddings_all_list, axis=0) if corpus_embeddings_all_list else np.empty((0,))

    if query_embeddings_file_path is not None:
        # Load query embeddings
        query_embeddings_file = f"{query_embeddings_file_path}/queries.pkl"
        if not os.path.exists(query_embeddings_file):
            logger.warning(f"Query embeddings file not found: {query_embeddings_file}")
        else:
            query_embeddings, query_ids = pickle_load(query_embeddings_file)

    return corpus_embeddings, all_corpus_ids, query_embeddings, query_ids

def Faiss_Search(corpus_embeddings, query_embeddings, corpus_ids, query_ids, batch_size=16, top_k=1000):
    """
    Perform FAISS-based similarity search.
    
    Args:
        corpus_embeddings: Corpus embeddings array
        query_embeddings: Query embeddings array  
        corpus_ids: List of corpus document IDs
        query_ids: List of query IDs
        batch_size: Batch size for search
        top_k: Number of top results to return
    
    Returns:
        dict: Results mapping query_id -> {corpus_id: score}
    """
    import gc
    
    # Create a fresh FAISS index on every call (local variable, auto-released on return)
    faiss_flat = FaissFlatSearcher(corpus_embeddings)
    faiss_flat.add(corpus_embeddings)  # Constructor only initialises structure; data must be added explicitly

    gpu_res_list = []  # Track all GPU resources (supports multi-GPU)
    gpu_index = None   # Hold a reference to the GPU index
    use_gpu = False
    
    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        logger.info("No GPU found or using faiss-cpu. Back to CPU.")
    else:
        logger.info(f"Using {num_gpus} GPU")
        try:
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                gpu_res = faiss.StandardGpuResources()
                # Disable pre-allocated temp memory pool; use on-demand allocation (easier to release)
                gpu_res.noTempMemory()
                gpu_res_list.append(gpu_res)
                gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_flat.index, co)
                faiss_flat.index = gpu_index
                use_gpu = True
            else:
                # Use index_cpu_to_all_gpus for multi-GPU (simpler and better compatibility)
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True  # Shard the index across multiple GPUs
                
                gpu_index = faiss.index_cpu_to_all_gpus(faiss_flat.index, co=co)
                faiss_flat.index = gpu_index
                use_gpu = True
        except Exception as e:
            logger.error(f"Error converting index to GPU: {e}. Falling back to CPU.")
            use_gpu = False
            # Clean up any resources already created
            for res in gpu_res_list:
                del res
            gpu_res_list.clear()

    try:
        scores, retrieved_indices = faiss_flat.batch_search(
            query_embeddings, top_k, batch_size=batch_size, quiet=False
        )
        retrieved_corpus_ids = [[str(corpus_ids[x]) for x in q_indices] for q_indices in retrieved_indices]

        results = {}
        for qid, doc_ids, score in zip(query_ids, retrieved_corpus_ids, scores):
            results[qid] = {doc_id: s for doc_id, s in zip(doc_ids, score)}
        return results
    finally:
        # ===== Explicitly release GPU memory (order matters) =====

        # Step 1: Synchronise all GPUs to ensure all operations are complete
        if use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Step 2: Delete GPU index first (must happen before releasing resources)
        if gpu_index is not None:
            del gpu_index

        # Step 3: Clear the reference held inside faiss_flat
        if hasattr(faiss_flat, 'index') and faiss_flat.index is not None:
            faiss_flat.index = None
        del faiss_flat

        # Step 4: Force GC to trigger C++ destructors
        gc.collect()

        # Step 5: Release GPU resources (after index has been deleted)
        for res in gpu_res_list:
            del res
        gpu_res_list.clear()

        # Step 6: GC again and clear the CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def embedding_transform(original_corpus_embeddings, original_query_embeddings, transform_type='pca',target_dim=768, kneedle_S=0.5, gamma=0.5, seed=2026):
    # Inputs and outputs are numpy arrays
    
    
    if transform_type == 'prefix_truncation':
        query_embeddings, corpus_embeddings = matryoshka_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim)
    
    
    elif transform_type == 'random_truncation':
        query_embeddings, corpus_embeddings = random_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, seed=seed)
   

    elif transform_type == 'pca':
        query_embeddings, corpus_embeddings = pca_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, seed=seed) 


    elif transform_type == 'whitening':
        query_embeddings, corpus_embeddings = whitening_k_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, seed=seed)

    elif transform_type == 'random_projection':
        query_embeddings, corpus_embeddings = random_projection_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, seed=seed)

    elif transform_type == 'spectemp':
        query_embeddings, corpus_embeddings = unified_spectral_tempering_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, auto_gamma=True, kneedle_S=kneedle_S, seed=seed)

    elif transform_type == 'y-whitening':
        query_embeddings, corpus_embeddings = unified_spectral_tempering_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, gamma=0.5, auto_gamma=False, kneedle_S=kneedle_S, seed=seed) # 0.5 is the default gamma here for y-whitening
    
    elif transform_type == 'grid_y-whitening':
        query_embeddings, corpus_embeddings = unified_spectral_tempering_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, gamma=gamma, auto_gamma=False, kneedle_S=kneedle_S, seed=seed) # gamma is the user-defined gamma here

    elif transform_type == 'none':
        corpus_embeddings = original_corpus_embeddings
        query_embeddings = original_query_embeddings
    else:
        raise ValueError(f"Invalid transform type: {transform_type}")
    
    return corpus_embeddings, query_embeddings

def config_parse():
    # Parse command line arguments for evaluation
    parser = argparse.ArgumentParser(description="Evaluate BEIR retrieval results with different LLM-based IR models.")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset to evaluate on (e.g., 'nq', 'msmarco' arguana nfcorpus scifact scidocs fiqa trec-covid)" )
    parser.add_argument("--model_name", type=str, default="gte", help="Path to the evaluation model. gte qwen3 diver jina_v4 ")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use for evaluation (default: 'test').")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=32, help="Batch size for evaluation (default: 256).")
    parser.add_argument('--seed', type=int, default=2026, help='Seed for evaluation.')
    parser.add_argument("--transform_type", type=str, default="spectemp", help="Transform type for the transform. none or pca,whitening, prefix_truncation, random_truncation, random_projection, ppa_pca_ppa spectemp y-whitening")
    parser.add_argument("--target_dim", type=int, default=768, help="Target dimension for the transform.")
    parser.add_argument("--kneedle_S", type=float, default=0.5, help="Kneedle algorithm sensitivity parameter.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Grid Y-Whitening gamma parameter.")

    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assumes the script is executed from the project root directory


    DEFAULT_EMBEDDING_ROOT = './embeddings/clean'
    DEFAULT_SCORES_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'output')

    embedding_save_root = DEFAULT_EMBEDDING_ROOT
    scores_save_root = DEFAULT_SCORES_ROOT

    config  =  config_parse()
    set_seed(config.seed)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout
    logging.info(config)


    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="Retrieval_eval_SpecTemp", 
        # track hyperparameters and run metadata
        config=vars(config),
    )

    # Build the embedding path; 'output/' prefix is kept for compatibility with the embedding save script
    embedding_save_path = os.path.join(
        embedding_save_root,
        f'output/embeddings/{config.model_name}/{config.dataset}'
    )
    os.makedirs(embedding_save_path, exist_ok=True)

    print(f"config: {config}")
    #### load eval_dataset
    # split: 'test' or 'dev', defaults to 'test'
    data_output_dic = load_beir_data(config.dataset ,split=config.split)
    corpus, queries, qrels = data_output_dic['corpus'], data_output_dic['queries'], data_output_dic['qrels']
    queries_raw = data_output_dic['queries_raw'] if 'queries_raw' in data_output_dic else None


    # Load saved embeddings and run retrieval
    top_k = 1000
    batch_size = config.per_gpu_eval_batch_size

    original_corpus_embeddings, all_corpus_ids, original_query_embeddings, query_ids = load_local_embeddings(embedding_save_path,embedding_save_path)
    
    print("original_corpus_embeddings.shape: ",original_corpus_embeddings.shape, "original_query_embeddings.shape: ",original_query_embeddings.shape)

    changed_corpus_embeddings, changed_query_embeddings = embedding_transform(original_corpus_embeddings, original_query_embeddings, transform_type=config.transform_type, target_dim=config.target_dim, kneedle_S=config.kneedle_S, gamma=config.gamma, seed=config.seed)


    print('changed_corpus_embeddings.shape: ',changed_corpus_embeddings.shape, 'changed_query_embeddings.shape: ',changed_query_embeddings.shape)
    changed_results = Faiss_Search(changed_corpus_embeddings, changed_query_embeddings, all_corpus_ids, query_ids,)

    # results = retriever.encode_and_retrieve(corpus, queries, encode_output_path = embedding_save_path, overwrite=False,)
    # results = retriever.retrieve(corpus, queries)

    output_list = _evaluate(changed_results, qrels, config, scores_save_root, queries_raw)

    # Prepare wandb summary log (all splits recorded in a single entry)
    wandb_log_dict = {}

    for output_dic in output_list:
        split = output_dic['split']
        output_all_scores = output_dic['output_all_scores']
        t_test_result = output_dic.get('t_test_result', None)

        # Log metrics for each split using the split name as prefix
        wandb_log_dict[f'{split}/NDCG@10'] = output_all_scores.get('NDCG@10', None)
        wandb_log_dict[f'{split}/MRR@10'] = output_all_scores.get('MRR@10', None)

        # Add t-test results if available
        if t_test_result is not None:
            wandb_log_dict[f'{split}/t_statistic'] = t_test_result['t_stat']
            wandb_log_dict[f'{split}/p_value'] = t_test_result['p_value']
            wandb_log_dict[f'{split}/is_significant'] = t_test_result['is_significant']
            wandb_log_dict[f'{split}/is_better'] = t_test_result['is_better']

        print(f"Prepared wandb log for split: {split}")

    # Log everything in a single wandb call
    wandb.log(wandb_log_dict)
    print(f"Logged all splits to wandb: {list(set([k.split('/')[0] for k in wandb_log_dict.keys() if '/' in k]))}")


def _evaluate(results, qrels, config, scores_save_root, queries_raw=None):

    # For BRIGHT datasets, remove excluded_ids from results
    if config.dataset in ["biology","earth_science","economics","psychology","robotics", "stackoverflow","sustainable_living","leetcode","pony","aops","theoremqa_theorems","theoremqa_questions"  ]:
        results = bright_scores_remove_excluded_ids(queries_raw,results)
        # Required for the non-StackExchange BRIGHT subsets

    if config.dataset == 'arguana':
        results = remove_identical_ids(results)
        
    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]

    if 'msmarco' in config.dataset:
        splits = ['dev' ,'trec_dl19' ,'trec_dl20' ,]
    elif 'browsecomp_plus' in config.dataset:
        splits = ['golds','evidence']
    else:
        splits = [config.split]
        qrels = {config.split: qrels}  # required: wrap qrels in a dict keyed by split

    output_list=[]
    for split in splits:
        output_dic={}
        output_dic['split']=split

        # Build transform_suffix encoding key hyperparams to avoid directory name collisions
        # The "none" baseline is seed-agnostic; always written to the shared 'none' directory
        if config.transform_type == 'none':
            transform_suffix = 'none'
        elif config.transform_type in ['spectemp', 'y-whitening', 'grid_y-whitening']:
            # These three transforms all use kneedle_S
            base_suffix = f'{config.transform_type}_{config.target_dim}'
            if config.transform_type == 'grid_y-whitening':
                # grid_y-whitening also depends on gamma
                base_suffix += f'_gamma{config.gamma}'
            base_suffix += f'_S{config.kneedle_S}'
            transform_suffix = f'{base_suffix}_seed{config.seed}'
        else:
            transform_suffix = f'{config.transform_type}_{config.target_dim}_seed{config.seed}'
        scores_save_path = os.path.join(
            scores_save_root,
            f'scores/{config.model_name}/{config.dataset}_{split}/{transform_suffix}/'
        )
        if not os.path.exists(scores_save_path):
            os.makedirs(scores_save_path)

        print(f"--------------------------------split: {split}--------------------------------")
        print(f"dataset: {config.dataset}, model_name: {config.model_name}, transform_type: {config.transform_type}, target_dim: {config.target_dim}")


        # Compute retrieval metrics
        output_all_scores, merged_query_level_scores = calculate_retrieval_metrics(
            results=results, qrels=qrels[split], return_scores=True
        )
        output_dic['output_all_scores']=output_all_scores
        output_dic['merged_query_level_scores']=merged_query_level_scores
        output_list.append(output_dic)

 

        # If transform is not "none", run a paired t-test against the "none" baseline
        # Use MRR@10 for msmarco, ndcg_cut_10 for all others
        if config.dataset == 'msmarco':
            metric = 'MRR@10'
        else:
            metric = 'ndcg_cut_10'
        if config.transform_type != 'none':
            # Load the "none" baseline scores from the shared 'none' directory (seed-agnostic)
            none_scores_path = os.path.join(
                scores_save_root,
                f'scores/{config.model_name}/{config.dataset}_{split}/none/merged_scores.json'
            )
            if not os.path.exists(none_scores_path):
                raise FileNotFoundError(f'none_scores_path: {none_scores_path} does not exist')
            with open(none_scores_path, 'r') as f:
                none_scores = json.load(f)
            # none_scores = {query_id: { metric: score}} ndcg_cut_10 MRR@10  

            scores_before, scores_after = [], []
            for query_id in none_scores:
                if query_id not in merged_query_level_scores:
                    print(f"Warning: {query_id} not found in merged_query_level_scores, skipping")
                    continue
                scores_before.append(none_scores[query_id][metric])
                scores_after.append(merged_query_level_scores[query_id][metric])
            print(f"scores_before length: {len(scores_before)}, scores_after length: {len(scores_after)}")
            
            t_stat, p_value = stats.ttest_rel(scores_before, scores_after)
            print(f"t-statistic: {t_stat}, p-value: {p_value}")

            alpha = 0.05

            is_significant = p_value < alpha
            is_better = t_stat < 0  # t_stat < 0 means after > before
            
            if is_significant:
                if is_better:
                    print(f"Result is significantly BETTER after {config.transform_type} transformation (p={p_value:.4f}).")
                else:
                    print(f"Result is significantly WORSE after {config.transform_type} transformation (p={p_value:.4f}).")
            else:
                print(f"The results are not statistically significant (p={p_value:.4f}).")
            
            # Save t-test result (cast to Python native types to ensure JSON serialisability)
            output_dic['t_test_result'] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'is_significant': bool(is_significant),
                'is_better': bool(is_better),
            } 
        
        
        # Save query-level scores
        with open(f'{scores_save_path}/merged_scores.json', 'w') as f:
            json.dump(merged_query_level_scores, f)
        # Merge t_test_result into output_all_scores if present
        if 't_test_result' in output_dic:
            output_all_scores.update(output_dic['t_test_result'])
        # Save aggregate metric summary
        with open(f'{scores_save_path}/output_all_scores.json', 'w') as f:
            json.dump(output_all_scores, f)
        # Save retrieval results
        with open(f'{scores_save_path}/retrieval_results.json', 'w') as f:
            json.dump(results, f)

        print('-------------------------------- the end of this run --------------------------------')
 

    return output_list

if __name__ == '__main__':
    main()