# premium_dl_ct
Deep learning on CT imaging to predict response to checkpoint inhibitors in melanoma


# todo
- adapt dataset to allow patient level classification tasks
    - only filter on missing lesion labels in train dataset, not in val and test
- implement 2.5d preprocessing
- implement different scales in preprocessing (both zoom and crop)
- data augmentation 
- cosine annealing and early stopping callbacks
- make model dependent on config dictionary
- random search over good initial parameters for model