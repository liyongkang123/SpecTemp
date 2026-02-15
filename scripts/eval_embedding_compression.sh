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
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate ir

nvidia-smi

cd /SpecTemp

model_list=(
    "qwen3" 
    "gte"     
    "embeddinggemma"
    "jina_v4"  
    "nomic_v2"
    "bge_m3"
)

dataset_list=(
    "msmarco"
    "nq"
    "fiqa"
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
    # "none" # "none" must be run separately first, as the baseline for subsequent paired t-tests
    "prefix_truncation"
    "random_truncation"
    "random_projection"
    "pca"
    "whitening"
    "y-whitening"
    "spectemp"
)

seed_list=(
    1999
    5
    2026 # default seed 
)

# Step 1: Run the "none" baseline first; no target_dim or transform argument is needed

for model in "${model_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do
        sbatch /SpecTemp/scripts/eval_embedding_compression_sub.sh $model $dataset "none" 512
    done
done


# Step 2: Run all other transforms; target_dim, transform, and seed arguments are required

# for model in "${model_list[@]}"; do
#     for dataset in "${dataset_list[@]}"; do
#         for target_dim in "${target_dims[@]}"; do
#             for transform in "${transform_list[@]}"; do
#                 for seed in "${seed_list[@]}"; do
#                     sbatch /SpecTemp/scripts/eval_embedding_compression_sub.sh "$model" "$dataset" "$transform" "$target_dim" "$seed"
#                 done
#             done
#         done
#     done
# done