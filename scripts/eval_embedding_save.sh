# 这个脚本的作用在于 评估从checkpoint中加载的模型的embedding性能，并将embedding保存到指定路径，避免每次评估都要重新计算embedding，节省时间


# embedding save 路径是 DEFAULT_EMBEDDING_ROOT ， 需要替换成您自己的

model_list=(
 
    "bge_m3"
    "qwen3"
    "gte"
    "embeddinggemma"
    "jina_v4" # jina_v4  的arugana 数据集需要设置 task 为 text-matching 才能取得 论文报告的结果，其他数据集下默认为 retrieval 任务
    "nomic_v2"
)     

 
dataset_list=(
    "msmarco"
    "nq"
    "fiqa"
    "fever"
)    

for model in "${model_list[@]}"
 for dataset in "${dataset_list[@]}"
  python eval_emebedding_save.py --model_name_or_path $model --dataset $dataset
done
    echo "Evaluating model: $model on dataset: $dataset"
done