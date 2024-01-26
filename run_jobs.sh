#!/bin/bash

# Define tasks, models, and their corresponding languages
declare -A tasks_models_langs=(
    ["xnli,bert-base-multilingual-cased"]="en ar de el es hi ru sw tr vi zh"
    ["xnli,xlm-roberta-base"]="en ar de el es hi ru sw tr vi zh"
    ["paws-x,bert-base-multilingual-cased"]="en de es fr ja ko zh"
    ["paws-x,xlm-roberta-base"]="en de es ja zh"
    ["xcopa,bert-base-multilingual-cased"]="en" # et ht id qu sw vi zh"
    ["xcopa,xlm-roberta-base"]="en" # et ht id it qu sw ta th tr vi zh"
)


# Define epochs for each task and data setup
declare -A epochs=(
    ["paws-x,all"]=6
    ["xnli,all"]=3
    ["xcopa,all"]=6
    ["paws-x,low-res"]=6
    ["xnli,low-res"]=6
    ["xcopa,low-res"]=6
)


# Seeds to use
seeds=(1 2 3 4 5)

# Manually set data setup
DATA_SETUP="all"  # Replace 'your_choice_here' with either 'all' or 'low-res'

# Iterate over each task-model pair
for key in "${!tasks_models_langs[@]}"; do
    IFS=',' read -ra task_model <<< "$key"
    task=${task_model[0]}
    model=${task_model[1]}
    train_langs=(${tasks_models_langs[$key]})

    # Determine epochs based on task and data setup
    epoch_key="${task},${DATA_SETUP}"
    num_epochs=${epochs[$epoch_key]}

    # Submit a job for each language and seed
    for lang in "${train_langs[@]}"; do
        for seed in "${seeds[@]}"; do
            sbatch ./train_sbatch.sh $task $DATA_SETUP $lang $model true $seed $num_epochs
        done
    done
done
