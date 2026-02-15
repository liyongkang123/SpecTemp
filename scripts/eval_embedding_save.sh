# This script evaluates embedding performance of models and saves embeddings to a specified path,
# avoiding recomputation on each run and saving time.


# The embedding save path is DEFAULT_EMBEDDING_ROOT, './embeddings/clean' - replace with your own path

model_list=(

    "bge_m3"
    "qwen3"
    "gte"
    "embeddinggemma"
    "jina_v4" # For jina_v4, the arguana dataset requires task='text-matching' to reproduce paper results; other datasets default to retrieval task
    "nomic_v2"
)

dataset_list=(
    "msmarco"
    "nq"
    "fiqa"
    "fever"
)

for model in "${model_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do
        echo "Evaluating model: $model on dataset: $dataset"
        python eval_emebedding_save.py --model_name $model --dataset $dataset
    done
done
