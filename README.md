# premium_dl_ct
Deep learning on CT imaging to predict response to checkpoint inhibitors in melanoma


# todo
- adapt dataset to allow patient level classification tasks
    - only filter on missing lesion labels in train dataset, not in val and test
- make model dependent on config dictionary
- cosine annealing and early stopping callbacks
- random search over good initial parameters for model
- inner cross validation
- run grid search
- inference using ensembling per inner fold
