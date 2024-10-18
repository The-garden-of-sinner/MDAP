
# Please go to the master branch

# Implement of WISE submission
[Paper Release](https://arxiv.org/abs/2410.05877)

Our model file is in `recbole_cdr/model/cross_domain_recommender/mdap.py`

## Datasets
[In Recbole](https://recbole.io/dataset_list.html)


## Requirements

```
recbole==1.0.1
torch>=1.7.0
python>=3.7.0
```

## Reproduce reported results 


1. Change config in `recbole_cdr/properties/model/MDAP.yaml` and `config/[dataset].yaml`

2. Run following 
```bash
python run_recbole_cdr.py --model=MDAP --config_files=./config/epinions.yaml --gpu_id=1

```

