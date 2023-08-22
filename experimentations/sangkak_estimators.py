
from pathlib import Path
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin
)
from pandas import DataFrame as pd_DataFrame

from features import read_pos_format_data, sent2labels, sent2features


__all__ = ['SangkakPosProjetReader', 'SangkakPosFeaturisation']


class SangkakPosProjetReader(TransformerMixin, BaseEstimator):

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, file_path, y=None):  
        self.file_path = file_path
        if Path(file_path).exists():
            self.file_path = file_path
            return self
        else: raise Exception("File not found: %s" % file_path)

    def transform(self, filename, copy=None, label=False):
        # check_is_fitted(self)
        _, pd_iob_data = read_pos_format_data(filename)
        
        return pd_iob_data
    
    def transform_analysis(self):
        # check_is_fitted(self)
        extracted_data, pd_iob_data = read_pos_format_data(self.file_path)
        
        return extracted_data, pd_iob_data

    def _more_tags(self):
        return {"stateless": True}


class SangkakPosFeaturisation(OneToOneFeatureMixin, TransformerMixin, 
                              BaseEstimator):

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):  
        return self

    def transform(self, X, copy=None, label=False):
        # check_is_fitted(self)
        
        copy = copy if copy is not None else self.copy
        # X = self._validate_data(X, accept_sparse="csr", reset=False)
        if copy:
            from copy import deepcopy
            X_cp = deepcopy(X)
        else: X_cp = X

        data_sents_formated = [[word for word in sentence] for sentence in X_cp]
       
        if label:
            X_cp = [sent2labels(s) for s in data_sents_formated]
        else:
            X_cp = [sent2features(s) for s in data_sents_formated]
        return X_cp


    def transform_to_sagemaker_format(self, Xt, yt, label="train",
                                      normalize=None):
        print(f"[{label}] Building sagemaker data for classification")            
        columns = ['labels']
        columns = columns + list(Xt[0][0].keys())
        df = pd_DataFrame(columns=columns)
        i = 0
        for x, y in zip(Xt, yt):
            for v, k in zip(x, y):
                row = [k]
                row = row + list(v.values())
                df.loc[i] = row
                i += 1
                
        if normalize is not None and type(normalize) == dict:
            categorical = normalize['categorical_features']
            numerical = normalize['numerical_features']
            df[categorical] = df[categorical].astype("category")
            assert len(list(df.select_dtypes(include="category").columns)) == len(categorical)

            df[numerical] = df[numerical].astype("int32")
            assert len(list(df.select_dtypes(include="number").columns)) == len(numerical)

        return df

    def _more_tags(self):
        return {"stateless": True}