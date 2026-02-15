'''
在常规的生成embedding之后，我会修改embedding 然后再 用于 检索
'''
import time
import torch
import json
import random
from scipy import stats
from transformers import (
    set_seed,
    AutoTokenizer,
)
import gc
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers.trainer_utils import set_seed
import wandb
import argparse
from utils.load_model import load_model_hf
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
from utils.utils import get_model_prompts_tasks
from isotropy_evals.isoscore_torch import IsoScore
from isotropy_evals.existing_scores_torch import isotropy_partition_score_torch_svd,isotropy_cosine
import glob
from beir.retrieval.search.base import BaseSearch
from beir.retrieval.search.dense.faiss_index import FaissFlatSearcher
from beir.retrieval.search.dense.util import cos_sim, dot_score, pickle_load, save_embeddings
from itertools import chain
from utils.utils import calculate_retrieval_metrics,remove_identical_ids


from embedding_transforms.baselines_numpy import matryoshka_truncation,random_truncation,pca_truncation,whitening_k_truncation,pca_all_but_the_top_truncation,random_projection_truncation
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

def isotropy_scores(embeddings):

    # 随机选择最多100k个embedding
    if embeddings.shape[0] > 100000:
        embeddings = embeddings[np.random.choice(embeddings.shape[0], 100000, replace=False)]

    start_time = time.time()
    isotropy_partition_score = isotropy_partition_score_torch_svd(embeddings)
    print('isotropy_partition_score time: ', time.time() - start_time)
    start_time = time.time()
    isotropy_cosine_score = isotropy_cosine(embeddings)
    print('isotropy_cosine_score time: ', time.time() - start_time)
    start_time = time.time()
    isocore = IsoScore(embeddings)
    print('isocore time: ', time.time() - start_time)
    return isotropy_partition_score, isotropy_cosine_score, isocore

def load_local_embeddings(corpus_embeddings_file_path=None, query_embeddings_file_path=None):
    # 初始化返回变量为 None
    corpus_embeddings = None
    all_corpus_ids = None
    query_embeddings = None
    query_ids = None
    
    if corpus_embeddings_file_path is not None:
        # 加载 corpus embeddings    
        corpus_embeddings_files = glob.glob(f"{corpus_embeddings_file_path}/corpus.*.pkl")
        
        if not corpus_embeddings_files:
            logger.warning(f"No corpus embedding files found in {corpus_embeddings_file_path}")
        else:
            logger.info("Loading precomputed corpus embeddings...")
            # 稳定、可解释的顺序（按分片编号排序）
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
        # 加载 query embeddings
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
    
    # 每次调用创建全新的 FAISS 索引 (局部变量，函数结束后自动释放)
    faiss_flat = FaissFlatSearcher(corpus_embeddings)
    faiss_flat.add(corpus_embeddings)  # 构造函数只初始化结构，需要显式添加数据
    
    gpu_res_list = []  # 用于追踪所有 GPU 资源（支持多卡）
    gpu_index = None   # 保存 GPU index 的引用
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
                # 禁用预分配的临时内存池，改为按需分配（更容易释放）
                gpu_res.noTempMemory()
                gpu_res_list.append(gpu_res)
                gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_flat.index, co)
                faiss_flat.index = gpu_index
                use_gpu = True
            else:
                # 多卡时使用 index_cpu_to_all_gpus（更简单且兼容性更好）
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True  # 分片到多个 GPU
                
                gpu_index = faiss.index_cpu_to_all_gpus(faiss_flat.index, co=co)
                faiss_flat.index = gpu_index
                use_gpu = True
        except Exception as e:
            logger.error(f"Error converting index to GPU: {e}. Falling back to CPU.")
            use_gpu = False
            # 清理已创建的资源
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
        # ===== 显式释放 GPU 显存（顺序很重要）=====
        
        # Step 1: 同步所有 GPU，确保操作完成
        if use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Step 2: 先删除 GPU index（必须在 resources 之前）
        if gpu_index is not None:
            del gpu_index
        
        # Step 3: 清理 faiss_flat 中的引用
        if hasattr(faiss_flat, 'index') and faiss_flat.index is not None:
            faiss_flat.index = None
        del faiss_flat
        
        # Step 4: 强制 GC 让 C++ 析构函数执行
        gc.collect()
        
        # Step 5: 删除 GPU resources（在 index 删除后）
        for res in gpu_res_list:
            del res
        gpu_res_list.clear()
        
        # Step 6: 再次 GC + 清空 CUDA 缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def embedding_transform(original_corpus_embeddings, original_query_embeddings, transform_type='pca',target_dim=768, kneedle_S=0.5, gamma=0.5, seed=2026):
    # 输入的其实是 numpy array, 但是我们使用torch 函数的话非常快速高效，所以更好用torch 函数·

    
    
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

    elif transform_type == 'ppa_pca_ppa':
        query_embeddings, corpus_embeddings = pca_all_but_the_top_truncation(original_query_embeddings, original_corpus_embeddings, target_dim=target_dim, seed=seed)

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
    parser.add_argument("--model_name", type=str, default="gte", help="Path to the evaluation model. gte qwen3 diver jina_v4 有问题其他的可以正常运行")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use for evaluation (default: 'test').")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=128, help="Batch size for evaluation (default: 256).")
    parser.add_argument('--seed', type=int, default=2026, help='Seed for evaluation.')
    parser.add_argument("--transform_type", type=str, default="spectemp", help="Transform type for the transform. none or pca,whitening, prefix_truncation, random_truncation, random_projection, ppa_pca_ppa spectemp y-whitening")
    parser.add_argument("--target_dim", type=int, default=768, help="Target dimension for the transform.")
    # parser.add_argument("--temperature", type=float, default=0.58, help="Temperature for the my own retrieval method.")
    parser.add_argument("--kneedle_S", type=float, default=0.5, help="Kneedle algorithm sensitivity parameter.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Grid Y-Whitening gamma parameter.")

    args = parser.parse_args()
    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取项目根路径（假设 eval.py 或 main 执行在项目根目录）


    DEFAULT_EMBEDDING_ROOT = '/scratch-shared/yli4/LLM_Robust/Clean'
    DEFAULT_SCORES_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'output')

    embedding_save_root = DEFAULT_EMBEDDING_ROOT
    # project_root = '/scratch-shared/yli4/LLM_Robust' # 临时修改
    # embedding_save_root = '/scratch-shared/yli4/LLM_Robust' # 临时修改
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
        project="LLM_robust_eval_improve_embedding_new", # 改成了 new, 作为最终版
        # track hyperparameters and run metadata
        config=vars(config),
    )

    # 结合项目根路径构建绝对路径
    embedding_save_path = os.path.join(
        embedding_save_root,
        f'output/embeddings/{config.model_name}/{config.dataset}' # 这里需要保留 output/ 因为之前的代码需要这个路径
    )
    os.makedirs(embedding_save_path, exist_ok=True)

    print(f"config: {config}")
    #### load eval_dataset
    # split = 'test' or 'dev',默认是test
    data_output_dic = load_beir_data(config.dataset ,split=config.split)
    corpus, queries, qrels = data_output_dic['corpus'], data_output_dic['queries'], data_output_dic['qrels']
    queries_raw = data_output_dic['queries_raw'] if 'queries_raw' in data_output_dic else None


    # 然后手动读取embedding并进行检索
    top_k = 1000
    batch_size = config.per_gpu_eval_batch_size

    original_corpus_embeddings, all_corpus_ids, original_query_embeddings, query_ids = load_local_embeddings(embedding_save_path,embedding_save_path)
    
    print("original_corpus_embeddings.shape: ",original_corpus_embeddings.shape, "original_query_embeddings.shape: ",original_query_embeddings.shape)

    isotropy_partition_score_before, isotropy_cosine_score_before, isocore_before = isotropy_scores(original_corpus_embeddings)

    print('before corpus: isotropy_partition_score: ',isotropy_partition_score_before, 'isotropy_cosine_score: ',isotropy_cosine_score_before, 'isocore: ',isocore_before)
    
    # 强制释放 isotropy_scores 占用的 GPU 显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
    print(f"[DEBUG] GPU memory after isotropy_scores: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    changed_corpus_embeddings, changed_query_embeddings = embedding_transform(original_corpus_embeddings, original_query_embeddings, transform_type=config.transform_type, target_dim=config.target_dim, kneedle_S=config.kneedle_S, gamma=config.gamma, seed=config.seed)


    print('changed_corpus_embeddings.shape: ',changed_corpus_embeddings.shape, 'changed_query_embeddings.shape: ',changed_query_embeddings.shape)
    isotropy_partition_score_after, isotropy_cosine_score_after, isocore_after = isotropy_scores(changed_corpus_embeddings)
    print('after corpus: isotropy_partition_score: ',isotropy_partition_score_after, 'isotropy_cosine_score: ',isotropy_cosine_score_after, 'isocore: ',isocore_after)

    changed_results = Faiss_Search(changed_corpus_embeddings, changed_query_embeddings, all_corpus_ids, query_ids,)

    # results = retriever.encode_and_retrieve(corpus, queries, encode_output_path = embedding_save_path, overwrite=False,)
    # results = retriever.retrieve(corpus, queries)

    # 准备 isotropy scores 字典传递给 _evaluate
    isotropy_scores_dict = {
        'partition_before': isotropy_partition_score_before,
        'cosine_before': isotropy_cosine_score_before,
        'isocore_before': isocore_before,
        'partition_after': isotropy_partition_score_after,
        'cosine_after': isotropy_cosine_score_after,
        'isocore_after': isocore_after,
    }
    
    output_list = _evaluate(changed_results, qrels, config, scores_save_root, queries_raw, isotropy_scores_dict)

    # 准备 wandb 汇总日志（所有 split 的结果在一条记录中）
    wandb_log_dict = {
        # Isotropy scores（只需记录一次）
        'isotropy_partition_before': isotropy_scores_dict['partition_before'],
        'isotropy_cosine_before': isotropy_scores_dict['cosine_before'],
        'isocore_before': isotropy_scores_dict['isocore_before'],
        'isotropy_partition_after': isotropy_scores_dict['partition_after'],
        'isotropy_cosine_after': isotropy_scores_dict['cosine_after'],
        'isocore_after': isotropy_scores_dict['isocore_after'],
    }
    
    for output_dic in output_list:
        split = output_dic['split']
        output_all_scores = output_dic['output_all_scores']
        t_test_result = output_dic.get('t_test_result', None)
        
        # 使用 split 作为前缀记录每个 split 的指标
        wandb_log_dict[f'{split}/NDCG@10'] = output_all_scores.get('NDCG@10', None)
        wandb_log_dict[f'{split}/MRR@10'] = output_all_scores.get('MRR@10', None)
        
        # 添加 t-test 结果（如果存在）
        if t_test_result is not None:
            wandb_log_dict[f'{split}/t_statistic'] = t_test_result['t_stat']
            wandb_log_dict[f'{split}/p_value'] = t_test_result['p_value']
            wandb_log_dict[f'{split}/is_significant'] = t_test_result['is_significant']
            wandb_log_dict[f'{split}/is_better'] = t_test_result['is_better']
        
        print(f"Prepared wandb log for split: {split}")
    
    # 一次性记录所有结果
    wandb.log(wandb_log_dict)
    print(f"Logged all splits to wandb: {list(set([k.split('/')[0] for k in wandb_log_dict.keys() if '/' in k]))}")


def _evaluate(results, qrels, config, scores_save_root, queries_raw=None, isotropy_scores_dict=None):

    # 如果是bright数据集，则有 excluded_ids 需要从 results 中删除
    if config.dataset in ["biology","earth_science","economics","psychology","robotics", "stackoverflow","sustainable_living","leetcode","pony","aops","theoremqa_theorems","theoremqa_questions"  ]:
        results = bright_scores_remove_excluded_ids(queries_raw,results)
        # 其实非 StackExchange 的5个数据集需要这样做

    if config.dataset == 'arguana':
        results = remove_identical_ids(results)
        
    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]

    if 'msmarco' in config.dataset:
        splits = ['dev' ,'trec_dl19' ,'trec_dl20' ,]
    elif 'browsecomp_plus' in config.dataset:
        splits = ['golds','evidence']
    else:
        splits = [config.split]
        qrels = {config.split: qrels} # 这是必须的

    output_list=[]
    for split in splits:
        output_dic={}
        output_dic['split']=split

        # 构造 transform_suffix，尽量把关键超参数都编码进去，避免同名覆盖
        # 对于 baseline（none），不区分 seed，统一写在 none 目录下
        if config.transform_type == 'none':
            transform_suffix = 'none'
        elif config.transform_type in ['spectemp', 'y-whitening', 'grid_y-whitening']:
            # 这三种都会用到 kneedle_S
            base_suffix = f'{config.transform_type}_{config.target_dim}'
            if config.transform_type == 'grid_y-whitening':
                # grid_y-whitening 还依赖 gamma
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
        # if config.dataset=='arguana':
        #     ignore_identical_ids = True
        # else:
        #     ignore_identical_ids = False


        # 使用自己写的calculate_retrieval_metrics 函数来计算指标
        output_all_scores, merged_query_level_scores = calculate_retrieval_metrics(
            results=results, qrels=qrels[split], return_scores=True
        )
        output_dic['output_all_scores']=output_all_scores
        output_dic['merged_query_level_scores']=merged_query_level_scores
        output_list.append(output_dic)

        # ndcg, _map, recall, precision ,all_scores = retriever.evaluate(qrels[split], results, retriever.k_values,ignore_identical_ids=ignore_identical_ids,return_scores=True) # 这里有 ignore_identical_ids
        # # print(ndcg, _map, recall, precision)
        # mrr ,mrr_scores = retriever.evaluate_custom(qrels[split], results, retriever.k_values, metric="mrr" ,return_scores=True) # 这里没有ignore_identical_ids
        # print('mrr: ' ,mrr)
        # merged_scores = merge_beir_eval_scores(all_scores,mrr_scores)
 

        # 如果 config.transform_type != 'none', 则和 none 的结果进行pairwise t-test, 根据 dataset 的不同选择不同的 metric
        # msmarco 使用 MRR@10, 其他使用 ndcg_cut_10
        # 进行pairwise t-test
        if config.dataset == 'msmarco':
            metric = 'MRR@10'
        else:
            metric = 'ndcg_cut_10'
        if config.transform_type != 'none':
            # 读取 baseline（none）的结果：统一从 none 目录读取，不区分 seed
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
            
            # 保存 t-test 结果到 output_dic（转为 Python 原生类型，避免 numpy 类型无法 JSON 序列化）
            output_dic['t_test_result'] = {
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'is_significant': bool(is_significant),
                'is_better': bool(is_better),
            } 
        
        
        # 保存 query-level scores 结果
        with open(f'{scores_save_path}/merged_scores.json', 'w') as f:
            json.dump(merged_query_level_scores, f)
        # 如果存在 t_test_result，合并到 output_all_scores 中一起保存
        if 't_test_result' in output_dic:
            output_all_scores.update(output_dic['t_test_result'])
        # 保存整体指标汇总（output_all_scores）
        with open(f'{scores_save_path}/output_all_scores.json', 'w') as f:
            json.dump(output_all_scores, f)
        # 保存 retrieval results
        with open(f'{scores_save_path}/retrieval_results.json', 'w') as f:
            json.dump(results, f)

        print('-------------------------------- the end of this run --------------------------------')
 

    return output_list

# 保存检索的结果
if __name__ == '__main__':
    main()