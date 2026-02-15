#!/bin/sh
#SBATCH --job-name=eval_embedding_compression
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --partition=gpu_h100
#SBATCH --time=00-02:00:00
#SBATCH --output=logs/%x-%j.out
# Set-up the environment.

# Activate conda
# eval "$(/gpfs/home1/yli4/anaconda3/bin/conda shell.bash hook)"
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate ir

nvidia-smi

cd /gpfs/work4/0/prjs0928/Embedding_Isotropy

model_list=(
    "qwen3" 
    "gte"     
    # "embeddinggemma"
    "jina_v4"  
    # "nomic_v2"
    # "bge_m3"
)

dataset_list=(
    # "msmarco"
    # "hotpotqa"
    # "nq"
    # "fiqa"
    # 'quora'
    'fever'
)
target_dims=(
    64
    128
    256
    512
    768
)

transform_list=(
    # "none" # "none"需要单独放
    "pca"
    "whitening"
    "prefix_truncation"
    "random_truncation"
    "random_projection"
    # "ppa_pca_ppa"
    "spectemp"
    "y-whitening"
)

seed_list=(
    1999
    5
    2026 # default seed 
)

# 把 none 运行，不需要 输入 target_dim 和 transform

# for model in "${model_list[@]}"; do
#     for dataset in "${dataset_list[@]}"; do
#         sbatch /gpfs/work4/0/prjs0928/Embedding_Isotropy/scripts/eval_embedding_compression_sub.sh $model $dataset "none" 512
#     done
# done


for model in "${model_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do
        for target_dim in "${target_dims[@]}"; do
            for transform in "${transform_list[@]}"; do
                for seed in "${seed_list[@]}"; do
                    sbatch /gpfs/work4/0/prjs0928/Embedding_Isotropy/scripts/eval_embedding_compression_sub.sh "$model" "$dataset" "$transform" "$target_dim" "$seed"
                done
            done
        done
    done
done