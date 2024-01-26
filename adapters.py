from datasets import load_dataset
from transformers.adapters.composition import Stack
from transformers import AutoConfig, AutoAdapterModel
from transformers import AdapterConfig
from transformers import AutoTokenizer
from transformers import TrainingArguments, AdapterTrainer
from datasets import concatenate_datasets
import numpy as np
from transformers import EvalPrediction

import argparse

import random
import numpy as np
import torch

import os

os.system('echo $HF_DATASETS_CACHE')

parser = argparse.ArgumentParser(description='Run NLP model training and evaluation.')

parser.add_argument('--task', type=str, default='xnli', help='Task to run (xnli, paws-x, or xcopa)')
parser.add_argument('--data_setup', type=str, default='low-res', choices=['all', 'low-res'], help='Data setup to use')
parser.add_argument('--train_lang', type=str, default='de', help='Training language')
parser.add_argument('--eval_langs', type=str, nargs='+', default=None, help='Evaluation languages')
parser.add_argument('--base_model', type=str, default='bert-base-multilingual-cased', help='Base model to use')
parser.add_argument('--none_tr', type=bool, default=True, help='No language adapter at training time')
parser.add_argument('--seed', type=int, default=123, help='Seed for reproducibility')
parser.add_argument('--epochs', type=int, default=6, help='Seed for reproducibility')
parser.add_argument('--train_output_dir', type=str, default='./training_output', help='Output directory for training artifacts')
parser.add_argument('--eval_output_dir', type=str, default='./eval_output', help='Output directory for evaluation artifacts')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

print(args)


task = args.task
data_setup = args.data_setup
train_lang = args.train_lang
eval_langs = args.eval_langs
base_model = args.base_model
none_tr = args.none_tr
seed_value = args.seed
epochs = args.epochs
train_output_dir = args.train_output_dir
eval_output_dir = args.eval_output_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed for reproducibility
set_seed(seed_value)

if eval_langs is None:
    if task == 'xnli' and base_model == 'bert-base-multilingual-cased':
        eval_langs = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ru', 'sw', 'tr', 'vi', 'zh']
    if task == 'xnli' and base_model == 'xlm-roberta-base':
        eval_langs = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ru', 'sw', 'tr', 'vi', 'zh']

    if task == 'paws-x' and base_model == 'bert-base-multilingual-cased':
        eval_langs = ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh']
    if task == 'paws-x' and base_model == 'xlm-roberta-base':
        eval_langs = ['en', 'de', 'es', 'ja', 'zh']

    if task == 'xcopa' and base_model == 'bert-base-multilingual-cased':
        eval_langs = ['et', 'ht', 'id', 'qu', 'sw', 'tr', 'vi', 'zh', ]
    if task == 'xcopa'and base_model == 'xlm-roberta-base':
        eval_langs = ['et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh']


if task == 'xcopa' and train_lang == 'en':
    dataset = load_dataset("super_glue", "copa")
else:
    dataset = load_dataset(task, train_lang)
    

tokenizer = AutoTokenizer.from_pretrained(base_model)

# batch encoders are dataset-specific
if task == 'paws-x':
    def encode_batch(examples):
        all_encoded = {"input_ids": [], "attention_mask": []}
        for sentence1, sentence2 in zip(examples["sentence1"], examples["sentence2"]):
            encoded = tokenizer(
                sentence1,
                sentence2,
                max_length=60,
                truncation=True,
                padding="max_length"
            )
            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
        return all_encoded

if task == 'xnli':
    def encode_batch(examples):
        all_encoded = {"input_ids": [], "attention_mask": []}
        for sentence1, sentence2 in zip(examples["premise"], examples["hypothesis"]):
            encoded = tokenizer(
                sentence1,
                max_length=60,
                truncation=True,
                padding="max_length",
            )
            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
        return all_encoded

if task == 'xcopa':
    def encode_batch(examples):
        all_encoded = {"input_ids": [], "attention_mask": []}
        for premise, question, choice1, choice2 in zip(examples["premise"], examples["question"], examples["choice1"], examples["choice2"]):
            sentences_a = [premise + " " + question for _ in range(2)]
            sentences_b = [choice1, choice2]
            encoded = tokenizer(
                sentences_a,
                sentences_b,
                max_length=60,
                truncation=True,
                padding="max_length",
            )
            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
        return all_encoded

def preprocess_dataset(data):
    data = data.map(encode_batch, batched=True)
    data.rename_column("label", "labels")
    data.set_format(columns=["input_ids", "attention_mask", "label"])
    return data

dataset = preprocess_dataset(dataset)


config = AutoConfig.from_pretrained(
    base_model,
)

model = AutoAdapterModel.from_pretrained(
    base_model,
    config=config,
).to(device)


# Load the language adapters
lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
print(lang_adapter_config)
model.load_adapter(f"{train_lang}/wiki@ukp", config=lang_adapter_config)

# Add a new task adapter
model.add_adapter(task)

# Add a classification head for our target task
if task == 'paws-x':
    model.add_classification_head("paws-x", num_labels=2)
elif task == 'xnli':
    model.add_classification_head("xnli", num_labels=3)
elif task == 'xcopa':
    model.add_multiple_choice_head("xcopa", num_choices=2)
    

model.train_adapter([task])

if none_tr:
    model.active_adapters = task # only task adapter activated for training (setup none_tr)
else:
    model.active_adapters = Stack(train_lang, task) # Source language adapter also activated for training (all other setups)


training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=epochs,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=1000,
    output_dir=train_output_dir,
    overwrite_output_dir=True,
    remove_unused_columns=False,
)


if task == 'xcopa':
    train_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
elif data_setup == 'all':
    train_dataset = dataset["train"]
elif data_setup == 'low-res':
    if task == 'xnli':
        train_dataset = dataset["train"].shuffle(seed=42).select(range(0,5000))
    else:
        train_dataset = dataset["train"].shuffle(seed=42).select(range(0,500))

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()


def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def evaluate(data, lang):
    
    if not none_tr:
        
        # With target adapter
        model.active_adapters = Stack(lang, task)

        eval_trainer = AdapterTrainer(
            model=model,
            args=TrainingArguments(output_dir=eval_output_dir, remove_unused_columns=False,),
            eval_dataset=data["test"],
            compute_metrics=compute_accuracy,
        )
        print(lang, ': ', eval_trainer.evaluate())

        # With source adapter
        model.active_adapters = Stack(train_lang, task)

        eval_trainer = AdapterTrainer(
            model=model,
            args=TrainingArguments(output_dir=eval_output_dir, remove_unused_columns=False,),
            eval_dataset=data["test"],
            compute_metrics=compute_accuracy,
        )
        print(f'{train_lang}:', eval_trainer.evaluate())

    model.active_adapters = task
    
    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(output_dir=eval_output_dir, remove_unused_columns=False,),
        eval_dataset=data["test"],
        compute_metrics=compute_accuracy,
    )
    print('no adapter: ', eval_trainer.evaluate())


for lang in eval_langs:
    
    if not none_tr:
        model.load_adapter(f"{lang}/wiki@ukp", config=lang_adapter_config)
    
    dataset_eval = load_dataset(task, lang)
    dataset_eval = preprocess_dataset(dataset_eval)
    evaluate(dataset_eval, lang)