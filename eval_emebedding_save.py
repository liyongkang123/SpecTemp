"""
Evaluate the main module of BEIR retrieval results
Support dense retrieval methods
"""


from transformers.trainer_utils import set_seed
import wandb
 
import argparse
# from beir import util
from utils.load_model import load_model_hf
from utils.load_data import load_beir_data
from utils.logging import LoggingHandler
from typing import Dict, Any, Optional, List


from utils.beir_custom_evaluation import EvaluateRetrieval
 
from utils.beir_exact_search import DenseRetrievalExactSearch as DRES_GPU

from utils.beir_utils import DenseEncoderModel, SentenceEncoderModel,SentenceEncoderModel_Prompt
from utils.utils import replace_slash, get_last_element, merge_beir_eval_scores, to_numpy,bright_scores_remove_excluded_ids
 
import logging
import os
import json
import torch
from torch import nn
 
from utils.utils import get_model_prompts_tasks
 
from utils.utils import calculate_retrieval_metrics

#### Just some code to print debug information to standard output
# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
#### /print debug information to standard output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_identical_ids(results):
    popped = []
    for qid, rels in results.items():
        for pid in list(rels):
            if qid == pid:
                results[qid].pop(pid)
                popped.append(pid)
    return results

# Configure constants
class Config:
    BRIGHT_DATASETS = {
        "biology", "earth_science", "economics", "psychology", "robotics",
        "stackoverflow", "sustainable_living", "leetcode", "pony", "aops",
        "theoremqa_theorems", "theoremqa_questions"
    }
    BEIR_DATASETS = {
    "trec-covid",    "nfcorpus", "nq", "hotpotqa", "fiqa",  "arguana",
    "webis-touche2020",    "quora", "dbpedia-entity", "scidocs", "fever",
    "climate-fever",     "scifact",
    }

    MSMARCO_SPLITS = ['dev', 'trec_dl19', 'trec_dl20']
    BROWSECOMP_SPLITS = ['golds', 'evidence']

    DEFAULT_K_VALUES = [1, 5, 10, 50, 100, 1000]
    DEFAULT_EMBEDDING_ROOT = './embeddings/clean'
    DEFAULT_SCORES_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'output')
    WANDB_PROJECT = "Retrieval_eval"

def config_parse():
    # Parse command line arguments for evaluation
    parser = argparse.ArgumentParser(description="Evaluate BEIR retrieval results with different LLM-based IR models.")
    parser.add_argument("--dataset", type=str, default="fiqa", help="Dataset to evaluate on (e.g., 'nq', 'msmarco' arguana nfcorpus scifact scidocs fiqa trec-covid)" )
    parser.add_argument("--model_name", type=str, default="gte", help="Path to the evaluation model.   gte qwen3  bge_m3  jina_v4 embeddinggemma  nomic_v2 other models can be run normally")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use for evaluation (default: 'test').")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=32, help="Batch size for evaluation (default: 256).")
    parser.add_argument('--seed', type=int, default=2026, help='Seed for evaluation.')
    parser.add_argument("--embedding_root", type=str, default=Config.DEFAULT_EMBEDDING_ROOT,help="Embedding save root path"    )
    parser.add_argument("--scores_root", type=str, default=Config.DEFAULT_SCORES_ROOT,help="Scores save root path")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging" )
    args = parser.parse_args()
    return args

def setup_experiment(config: argparse.Namespace) -> Dict[str, Any]:
    """Setup experiment and path"""
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # Path settings
    # scores_save_root = os.path.abspath(os.path.dirname(__file__))
    scores_save_root = config.scores_root
    embedding_save_path = os.path.join(
        config.embedding_root,
        f'output/embeddings/{config.model_name}/{config.dataset}'
    )
    os.makedirs(embedding_save_path, exist_ok=True)

    # Initialize wandb
    wandb_run = None
    if not config.no_wandb:
        wandb_run = wandb.init(
            project= Config.WANDB_PROJECT,
            config=vars(config),
            name=f"{config.model_name}_{config.dataset}_{config.split}"
        )

    logger.info(f"Configuration: {config}")

    return {
        'device': device,
        'embedding_save_path': embedding_save_path,
        'scores_save_root': scores_save_root,
        'wandb_run': wandb_run
    }


def DenseRetrieval(config):
    # Setup experiment environment
    experiment_setup = setup_experiment(config)

    #### load eval_dataset
    # split = 'test' or 'dev', default is test
    data_output_dic = load_beir_data(config.dataset ,split=config.split)
    corpus, queries, qrels = data_output_dic['corpus'], data_output_dic['queries'], data_output_dic['qrels']
    queries_raw = data_output_dic['queries_raw'] if 'queries_raw' in data_output_dic else None

    #### Load the sentence-transformer model and retrieve using cosine-similarity
    prompts = get_model_prompts_tasks(model_name=config.model_name,dataset_name=config.dataset)
    print(f"prompts: {prompts}")

    encoder,tokenizer = load_model_hf(config.model_name)
    # model = DRES_GPU(SentenceEncoderModel(encoder, prompts), batch_size=config.per_gpu_eval_batch_size)
    # For large datasets, when using GPU, it is necessary to ensure that there are at least 2 GPUs, otherwise the memory will be insufficient and an error will be reported
    model = DRES_GPU(SentenceEncoderModel_Prompt(encoder, prompts, config.model_name), batch_size=config.per_gpu_eval_batch_size)

    score_function = "cos_sim" if config.model_name not in ['contriever', 'tas_b'] else "dot"
    
    retriever = EvaluateRetrieval(model, score_function=score_function, k_values=[1, 5, 10, 50 ,100, 1000] , ) # or "dot" for dot product cos_sim
    results = retriever.encode_and_retrieve(corpus, queries, encode_output_path = experiment_setup['embedding_save_path'], overwrite=False, score_function = score_function)
    # In fact, if encode_and_retrieve is used, the score_function set earlier is invalid, and it must be set in the model itself normalize=True
 
    # Process special datasets, if the dataset is bright, then excluded_ids need to be removed from results # In fact, non-StackExchange 5 datasets need to do this
    if config.dataset in Config.BRIGHT_DATASETS :
        results = bright_scores_remove_excluded_ids(queries_raw, results)
    
    if config.dataset == 'arguana':
        results = remove_identical_ids(results)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]

    if 'msmarco' in config.dataset:
        splits = ['dev' ,'trec_dl19' ,'trec_dl20' ,]
    elif 'browsecomp_plus' in config.dataset:
        splits = ['golds','evidence']
    else:
        splits = [config.split]
        qrels = {config.split: qrels}

    for split_name in splits:
        split_qrels = qrels[split_name]
        split_results = {qid: docs for qid, docs in results.items() if qid in split_qrels} # Only keep results in split_qrels
        _evaluate_split(split_name, split_results, split_qrels, config, experiment_setup['scores_save_root'], experiment_setup['wandb_run'])


def _evaluate_split(split, results, split_qrels, config, scores_save_root, wandb_logger=None):
    """Evaluate single data split"""
    scores_save_path = os.path.join(
        scores_save_root, f'scores/{config.model_name}/{config.dataset}_{split}/'
    )
    os.makedirs(scores_save_path, exist_ok=True)

    print(f"--------------------------------split: {split}--------------------------------")

    # Calculate retrieval metrics
    output_all_score, merged_query_level_scores = calculate_retrieval_metrics(
        results=results, qrels=split_qrels, return_scores=True
    )

    # Save results
    with open(f'{scores_save_path}/merged_scores.json', 'w') as f:
        json.dump(merged_query_level_scores, f)
    with open(f'{scores_save_path}/retrieval_results.json', 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {scores_save_path}")

    # Record to wandb (if logger is provided)
    if wandb_logger:
        wandb_logger.log({
            f"{split}_NDCG@10": output_all_score['NDCG@10'],
            f"{split}_Recall@100": output_all_score['Recall@100'],
            f"{split}_MRR@10": output_all_score['MRR@10'],
        })



# Save retrieval results
if __name__ == '__main__':
    config = config_parse()
    DenseRetrieval(config)
