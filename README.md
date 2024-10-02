# ThinkHub: Engage in Divergent Thinking with Implicit and Explicit Reasoning Memory

This is the code for the paper *ThinkHub: Engage in Divergent Thinking with Implicit and Explicit Reasoning Memory*.

## Getting Started

We recommand using conda environment to set up.


```
$ conda create -n thinkhub python=3.9
$ conda activate thinkhub
$ pip install -r requirements.txt
$ pip install -e .
```

Some code was borrowed from the following repositories:

NeMo-Skills: https://github.com/Kipok/NeMo-Skills


## Sample Data

```bash 
# logicqa is the dataset name, deductive is the reasoning type, 16 is the batch size
$ bash run_sample.sh logicqa deductive 16
```

## Finetune the model

```bash 
# sft is the model name, all is the data version
$ bash run_finetune.sh sft all
```

## Evalution

### Evalute baselines

```bash 
# fewshot is the baseline name, 1 is the number of responses, 60 is the batch size
$ bash run_baseline.sh fewshot 1 60
```

### Evalute ThinkHub


```bash 
# Get the reasoning type
$ python agent.py --func test --train_from [meta thinker] --step 450 --benchmark all 

# Apply the reasoning type
$ python agent.py --func apply --load_from [meta thinker] --step 450 --train_from [sft reasoner] --benchmark all --mode fewshot --policy_mode best --n 1
```

