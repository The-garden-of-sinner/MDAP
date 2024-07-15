
# Implement of WISE submission

Our model file is in `recbole_cdr/model/cross_domain_recommender/mdap.py`

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

