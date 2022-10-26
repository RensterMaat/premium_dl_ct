# premium_dl_ct
Deep learning on CT imaging to predict response to checkpoint inhibitors in melanoma


# todo
- adapt dataset to allow patient level classification tasks
    - add patient id to batch
    - make sampler to add patients to batch until batch is full
    - only filter on missing lesion labels in train dataset, not in val and test
    - add patient outcome to dataset, but also to datamodule? how to do this neatly?
- lesion and patient level auc as metric
- data augmentation 
- cosine annealing and early stopping callbacks
- implement 2.5d preprocessing
    - 
- implement different scales in preprocessing (both zoom and crop)
- make model dependent on config dictionary
- random search over good initial parameters for model