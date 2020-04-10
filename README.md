# NLPPerfPred
Performance Prediction for NLP Tasks

## Requirments 
pytorch==1.2.0

## Scripts

###  Representativeness
Search types include best_search, worst_search and random_search. n denotes the maximum size of the final set and beam_size denotes the number of expanded nodes in each step.
```bash
python3 representativeness.py --task monomt \ 
                              --log logs/monomt_bs.log \
                              --n 5 \
                              --beam_size 100 \
                              --type best_search
```

### New Models
Using existing experimental records for other models and n experimental records of 50% of the experimental records from a new coming model to make predictions on the rest of the 50% percent.  
```bash
n = 5
python3 new_model.py --task ud \
                     --log logs/ud_nm_${5}.log \
                     --n 5 \
                     --portion 0.5 \
                     --test_id_options_num 100 \
                     --sample_options_num 100
```
