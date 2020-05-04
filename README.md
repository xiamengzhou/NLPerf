# NLPerf 
**NLPerf** is an exploratory framework to evaluate performance for NLP tasks by training predictors using past experimental records. We provide data, code, experiment logs for 9 tasks in this repository. For mode technical
details, please refer to our paper:

[Predicting Performance for Natural Language Processing Tasks](google.com)

## Scripts

### Predicting Performance ###
K-fold evaluation for TED-MT over 10 random runs. You'll have to modify the parameters in the code.
```bash
python3 main_code.py 

...
The average rmse for 10 runs is 6.177170443398633
The average k_split baseline rmse for 10 runs is 12.653809064828215
The average Source Language baseline rmse for 10 runs is 12.653809064828215
The average Target Language baseline rmse for 10 runs is 12.653809064828215

```

For single model tasks:
```python
k_fold_evaluation("wiki",
                  shuffle=True,
                  selected_feats=None,
                  combine_models=False,
                  regressor="xgboost",
                  k=5,
                  num_running=10)
```

For multi model tasks:
```python
k_fold_evaluation("ud",
                  shuffle=True,
                  selected_feats=None,
                  combine_models=True,
                  regressor="xgboost",
                  k=5,
                  num_running=10)
```

You can specify a set of features for training by passing a list of feature names to selected_feats.

###  Representativeness
Search types include best_search, worst_search and random_search. n denotes the maximum size of the final set and beam_size denotes the number of expanded nodes in each step. Note that monomt denotes the TED-MT task. 
```bash
python3 representativeness.py --task monomt \ 
                              --log logs/monomt_bs.log \
                              --n 5 \
                              --beam_size 100 \
                              --type best_search
```

### New Models
Using existing experimental records for other models and n experimental records of 50% of the experimental records from a new coming model to make predictions on the rest of the 50%.  
```bash
n = 5
python3 new_model.py --task ud \
                     --log logs/ud_nm_${n}.log \
                     --n 5 \
                     --portion 0.5 \
                     --test_id_options_num 100 \
                     --sample_options_num 100
```
