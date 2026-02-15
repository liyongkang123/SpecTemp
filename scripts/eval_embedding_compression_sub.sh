#!/bin/sh
#SBATCH --job-name=eval_embedding_compression_sub 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=360G
#SBATCH -p gpu
#SBATCH --gres gpu:2
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

model=$1
dataset=$2
transform=$3
target_dim=$4
seed=$5

python eval_embedding_compression.py --model_name $model --dataset $dataset --transform_type $transform --target_dim $target_dim --seed $seed
