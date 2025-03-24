from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
# from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score,  precision_recall_curve, roc_curve, f1_score, precision_score, recall_score, balanced_accuracy_score

from sklearn.preprocessing import label_binarize


def score_predict(func):
    def new_func(y_true, y_pred,*args,**kwargs):
        y_pred = np.argmax(y_pred,axis =1)
        return func(y_true, y_pred,*args,**kwargs)
    return new_func
def score_predict_proba(func):
    def new_func(y_true, y_pred,*args,**kwargs):
        y_true = label_binarize(y_true,classes = list(np.sort(np.unique(y_true))))
        if y_true.shape[1]==1:
            y_true = np.concatenate([y_true^1,y_true],axis=1)
        y_true = np.argmax(y_pred,axis =1)
        return func(y_true, y_pred,*args,**kwargs)
    return new_func

SCORING_FUNC = {
    'balanced_accuracy': score_predict(balanced_accuracy_score),
    'roc_auc_ovr': lambda y_true, y_pred:roc_auc_score(y_true, y_pred if y_pred.shape[1]>2 else y_pred[:,1],multi_class="ovr") ,
    'average_precision': score_predict(average_precision_score),
    'f1_weighted': score_predict(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')),
    'precision_weighted': score_predict(lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted')),
    'recall_weighted': score_predict(lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted')),
    'TP':score_predict(lambda x,y:(x==y)[x!=0].sum()),
    'TN':score_predict(lambda x,y:(x==y)[x==0].sum()),
    'FP':score_predict(lambda x,y:(x!=y)[x==0].sum()),
    'FN':score_predict(lambda x,y:(x!=y)[x!=0].sum()),
    'count' : lambda y_true, y_pred : len(y_true)
}


SCORING = list(SCORING_FUNC.keys())

class QuantilNorm(TransformerMixin):
    def __init__(self,q1=0.01,q2=0.99):
        self.q1,self.q2=q1,q2
    def fit(self,X,y=None):
        self.Q1 = np.quantile(X,self.q1)
        self.Q2 = np.quantile(X,self.q2)
        return self
    def transform(self,X):
        return (X-self.Q1) / (self.Q2 - self.Q1)

class GausNorm(TransformerMixin):
    def fit(self,X,y=None):
        self.mean = X.mean()
        self.std= X.std()
        return self
    def transform(self,X):
        return (X-self.mean ) / self.std()
    
PREPROC = {
    'Quantil norm':QuantilNorm,
    'Gaussian norm':GausNorm,
    'PCA':PCA,
    'SMOTE':SMOTE,
    'RandomUnderSampler':RandomUnderSampler
}


class Classifier:
    simplified_args = {}
    parser = {}
    features_kwarg_str = {}
    transforms = []
    pipeline = None

    def pars_features_arg(self) -> dict:        
        return {key: self.parser[key](val) for key, val in self.features_kwarg_str.items()}
    
    def fit_pipe(self,X:pd.DataFrame,bag_names = []):
        X, Y, self.classes_to_id =  make_classes(X,bag_names)
        self.columns = X.columns
        self.index_names = X.index.names
        self.pipeline = Pipeline(steps = [(el,PREPROC[el]()) for el in self.transforms] + [('clf',self.clf)])
        self.pipeline.fit(X,Y)

    def pred_pipe(self,X:pd.DataFrame,bag_names = []):

        assert self.pipeline is not None, "model have to be fitted"
        X = X[self.columns]
        Z = self.pipeline.predict_proba(X)

        if len(self.index_names)>1:
            col_index = pd.MultiIndex.from_tuples(list(self.classes_to_id.keys()), names = self.index_names)
        else:
            col_index = pd.MultiIndex.from_tuples([(el,) for el in self.classes_to_id.keys()], names = self.index_names)
        return pd.DataFrame(Z, index = X.index, columns=col_index)

    def cross_validate(self,X:pd.DataFrame,bag_names = []):
        X, Y, self.classes_to_id =  make_classes(X,bag_names)
        self.columns = X.columns
        self.index_names = X.index.names
        self.pipeline = Pipeline(steps = [(el,PREPROC[el]()) for el in self.transforms] + [('clf',self.clf)])
        return cross_validate(self.pipeline, X, Y, scoring=SCORING, cv=5)

    def pred_pipe(self,X:pd.DataFrame,bag_names = []):

        assert self.pipeline is not None, "model have to be fitted"
        X = X[self.columns]
        Z = self.pipeline.predict_proba(X)

        if len(self.index_names)>1:
            col_index = pd.MultiIndex.from_tuples(list(self.classes_to_id.keys()), names = self.index_names)
        else:
            col_index = pd.MultiIndex.from_tuples([(el,) for el in self.classes_to_id.keys()], names = self.index_names)
        return pd.DataFrame(Z, index = X.index, columns=col_index)

    def test(self,X:pd.DataFrame,bag_names = []):
        assert self.pipeline is not None, "model have to be fitted"
        X, Y_True, _ =  make_classes(X,bag_names,self.classes_to_id)
        Y_Pred = self.pipeline.predict_proba(X)
        # if Y_Pred.shape[-1]==2:
        #     Y_Pred = Y_Pred[...,1]
        # Y_True = label_binarize(Y_True,classes = list(np.sort(np.unique(Y_True))))
        # if Y_True.shape[1]==1:
        #     Y_True = np.concatenate([Y_True^1,Y_True],axis=1)
        scores = {el:SCORING_FUNC[el](Y_True,Y_Pred) for el in SCORING}
        return scores

        # lb = LabelBinarizer()
        # lb.fit(Y)
        # acc = balanced_accuracy_score(Y,np.argmax(Z,axis = 1))
        # prec = average_precision_score(lb.transform(Y),Z if Z.shape[1]>2 else np.argmax(Z,axis = 1),average='samples')

        # return {'test_balanced_accuracy':acc, "test_average_precision":prec}

    
    def optimize_params(self,X:pd.DataFrame,n_iter=20,bag_names = []):
        X, Y, self.classes_to_id =  make_classes(X,bag_names)
        self.columns = X.columns
        self.index_names = X.index.names
        self.pipeline = Pipeline(steps = [(el,PREPROC[el]()) for el in self.transforms] + [('clf',self.clf)])
        grid_str = {k:[v for v in self.simplified_args[k]] for k in  self.simplified_args}
        [grid_str[k].append(el) for k,el in self.features_kwarg_str.items() if el not in grid_str[k]]
        try:
            grid = {("clf__")+k:[self.parser[k](v) for v in grid_str[k]] for k in  grid_str}
            search = RandomizedSearchCV(self.pipeline, grid, cv=5,n_jobs=16,n_iter=n_iter, scoring=SCORING,refit = SCORING[0],error_score=0.).fit(X, Y)
        except:
            grid = {('clf__clf__' )+k:[self.parser[k](v) for v in grid_str[k]] for k in  grid_str}
            search = RandomizedSearchCV(self.pipeline, grid, cv=5,n_jobs=16,n_iter=n_iter, scoring=SCORING,refit = SCORING[0],error_score=0.).fit(X, Y)
        bests:dict = search.best_params_
        for kpipe, el in bests.items():
            k=kpipe.replace('clf__','')
            for v in grid_str[k]:
                if self.parser[k](v)==el:
                    self.features_kwarg_str[k]=v
 
        self.clf = search.best_estimator_
        best_score = {}
        best_id = search.best_index_
        for k, el in search.cv_results_.items():
            if k.startswith("mean_"):
               best_score[k.replace("mean_",'').replace("test_",'')] = el[best_id]
        return best_score
    

def make_classes(X:pd.DataFrame, bag_names=[], classes_to_id = None):
    """Generate Int classes (0,1,2...n) fromm X index, and return the coresponding y column

    Args:
        X (pd.DataFrame): Input dataframe
        bag_names (list, optional): Index level to be excluded from classification
        classes_to_id (dict, optional): mapper between classes and index. Defaults to None.

    Returns:
        tuple:
        X,Y, classes_to_id
    """
    if len(bag_names)>0:
        X=X.droplevel(bag_names)

    X['Y'] = 0
    if classes_to_id is None:
        classes_to_id = {el:i for i, el in enumerate(X.index.unique())} 
    for k,el in classes_to_id.items():
        X.loc[k,"Y"] = el
    Y = X.pop("Y")
    return X, Y, classes_to_id


class MLP(Classifier):

    simplified_args = {
        'hidden_layer_sizes':['100,','50,50',"20,100,20","50,10,10,50"],
        'activation': ['relu', 'identity', 'logistic', 'tanh'],
        'learning_rate':['constant', 'invscaling', 'adaptive'],
        'max_iter':['200','1000'],
        'solver': ['adam', 'lbfgs', 'sgd']
    }
    parser = {
        'hidden_layer_sizes':lambda x: tuple([int(el) for el in x.split(',') if el !=""]),
        'activation': lambda x:x,
        'learning_rate':lambda x:x,
        'max_iter':lambda x:int(x),
        'solver': lambda x:x
    }

    def __init__(self, features_kwarg_str:dict = None):
        if features_kwarg_str is None:
            features_kwarg_str = {k:v[0] for k,v in self.simplified_args.items()}
        self.features_kwarg_str:dict = features_kwarg_str
        self.clf = MLPClassifier(**self.pars_features_arg())

class RandomForest(Classifier):

    simplified_args = {
        'n_estimators' : ["100","200","50","1000","10"],
        'max_depth': ["None","5","10","20"],
        'min_samples_split' : ["2","5","10"],
        'min_samples_leaf' : ["1","5","10"],
        'max_features' : ["auto", "sqrt", "log2", "10"],
    }

    parser = {
        'n_estimators' : lambda x:int(x),
        'max_depth': lambda x:int(x) if x!='None' else None,
        'min_samples_split' : lambda x:int(x) if float(x)>=1 else float(x),
        'min_samples_leaf' : lambda x:int(x) if float(x)>=1 else float(x),
        'max_features' : lambda x:int(x) if x not in ["auto", "sqrt", "log2"] else x ,
    }
    
    def __init__(self, features_kwarg_str:dict = None):
        if features_kwarg_str is None:
            features_kwarg_str = {k:v[0] for k,v in self.simplified_args.items()}
        self.features_kwarg_str:dict = features_kwarg_str
        self.clf = RandomForestClassifier(**self.pars_features_arg())

class DecisionTree(Classifier):

    simplified_args = {
        'max_depth': ["None","5","10","20"],
        'min_samples_split' : ["2","5"],
        'min_samples_leaf' : ["1","5"],
        'max_features' : ["auto", "sqrt", "log2", "10"],
    }

    parser = {
        'max_depth': lambda x:int(x) if x!='None' else None,
        'min_samples_split' : lambda x:int(x) if float(x)>=1 else float(x),
        'min_samples_leaf' : lambda x:int(x) if float(x)>=1 else float(x),
        'max_features' : lambda x:int(x) if x not in ["auto", "sqrt", "log2"] else x ,
    }
    
    def __init__(self, features_kwarg_str:dict = None):
        if features_kwarg_str is None:
            features_kwarg_str = {k:v[0] for k,v in self.simplified_args.items()}
        self.features_kwarg_str:dict = features_kwarg_str
        self.clf = DecisionTreeClassifier(**self.pars_features_arg())

class SVM(Classifier):

    simplified_args = {
        'C': ["1.0","2.0","5.0","1.0"],
        'degree' : ["3","4","5"],
        'gamma' : ["scale","auto","0.1"],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    }

    parser = {
        'C': lambda x:float(x),
        'degree' : lambda x:int(x) ,
        'gamma' : lambda x:x if x in ["scale","auto"] else float(x),
        'kernel' : lambda x:x ,
    }
    
    def __init__(self, features_kwarg_str:dict = None):
        if features_kwarg_str is None:
            features_kwarg_str = {k:v[0] for k,v in self.simplified_args.items()}
        self.features_kwarg_str:dict = features_kwarg_str
        self.clf = SVC(probability=True,**self.pars_features_arg())

MODEL_LIB = {"Default SVM":SVM(), 'Default Tree':DecisionTree(),"Default RF": RandomForest(), "Default MLP": MLP()}