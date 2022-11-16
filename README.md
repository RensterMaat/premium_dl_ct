# premium_dl_ct
Deep learning on CT imaging to predict response to checkpoint inhibitors in melanoma


# todo
- adapt dataset to allow patient level classification tasks
    - only filter on missing lesion labels in train dataset, not in val and test
- adaptive batch size
- random search over good initial parameters for model on dummy task, 1 fold
- start extra agent on hpc
- inner cross validation
- run grid search
- inference using ensembling per inner fold

# nice to have
- multiple instance learning
