# The Impact of Language Adapters in Cross-Lingual Transfer for NLU

This repository contains the code to reproduce the results in the paper [The Impact of Language Adapters in Cross-Lingual Transfer for NLU](https://openreview.net/pdf?id=cLcOAlYu8t).

## Setup

Required libraries and package dependencies are presented in adapters-env.def.

## Run example

```
python adapters.py \
    --task "xnli" \
    --data_setup "all" \
    --train_lang "en" \
    --base_model "xlm-roberta-base" \
    --seed 1 \
    --epochs 3 \
    --train_output_dir "results/training" \
    --eval_output_dir "results/eval"
```

We provide the files we used to batch jobs on our computer cluster.

## Citation

```

```

## Contact

For any questions or inquiries, please contact oskar.holmstrom@liu.se.
