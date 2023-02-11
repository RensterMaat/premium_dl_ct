import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from pathlib import Path
from data import CENTERS


class ClinicalModel:
    def __init__(self, predictions_save_folder):
        self.model = Pipeline([
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)),
            ('lr', LogisticRegression())
        ])
        self.grid_search = GridSearchCV(
            self.model,
            {'lr__C': np.logspace(-4,4,10)},
            scoring='roc_auc',
            n_jobs=1
        )
        self.predictions_save_folder = predictions_save_folder

    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.model.set_params(**self.grid_search.best_params_)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

        training_set_predictions = pd.Series(
            cross_val_predict(self.model, X, y, method='predict_proba')[:,1],
            index=X.index
        )
        training_set_predictions.to_csv(self.predictions_save_folder / 'clinical_training_preds.csv')

    def predict_proba(self, X):
        preds = self.model.predict_proba(X)
        pd.Series(preds[:,1], index=X.index).to_csv(self.predictions_save_folder / 'clinical_test_preds.csv')
        return preds

    def get_params(self, *args, **kwargs):
        return {}

    def set_params(self, *args, **kwargs):
        return 
        

if __name__ == "__main__":
    clinical_predictors = pd.read_csv('/mnt/c/Users/user/data/tables/clinical_predictors.csv').set_index('id')

    save_folder = Path('/mnt/c/Users/user/data/results_dl')
    model = ClinicalModel(None)

    for center in CENTERS:
        model.predictions_save_folder = save_folder / center

        train = clinical_predictors[clinical_predictors.center != center]
        X_train = train.drop(columns=['center','response'])
        y_train = train.response
        model.fit(X_train, y_train)

        test = clinical_predictors[clinical_predictors.center == center]
        X_test = test.drop(columns=['center','response'])
        model.predict_proba(X_test)