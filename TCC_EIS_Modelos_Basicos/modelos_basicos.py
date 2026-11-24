"""Modelos básicos, funções de validação e classificador simbólico DIDÁTICO.

Atenção: SymbolicOneTermClassifier NÃO é implementação oficial do SISSO.
Ele usa triagem de poucas expressões e regressão logística de 1 termo.
"""
from collections import OrderedDict
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import accuracy_score,balanced_accuracy_score, confusion_matrix

class SymbolicOneTermClassifier(BaseEstimator,ClassifierMixin):
    """Modelo esparso exploratório inspirado em SISSO, **não** SISSO oficial.

    Constrói atributos candidatos simples em cada *fit*, seleciona 1 com maior
    correlação absoluta com o alvo apenas no treino e ajusta regressão logística.
    O teste não participa da triagem nem do ajuste.
    """
    def __init__(self, C=1.0):
        self.C=C
    @staticmethod
    def _expand(x, names=False):
        x=np.asarray(x,dtype=float)
        a=[x[:,i] for i in range(x.shape[1])]
        nm=[f'x{i+1}' for i in range(x.shape[1])]
        for i in range(x.shape[1]):
            for j in range(i+1,x.shape[1]):
                a.extend([x[:,i]+x[:,j],x[:,i]-x[:,j],x[:,i]*x[:,j]])
                nm.extend([f'x{i+1}+x{j+1}',f'x{i+1}-x{j+1}',f'x{i+1}*x{j+1}'])
        return nm if names else np.column_stack(a)
    def fit(self,X,y):
        X=np.asarray(X,dtype=float)
        y=np.asarray(y,dtype=int)
        self.classes_=np.unique(y)
        if len(self.classes_)!=2: raise ValueError('Modelo exige as duas classes no treino.')
        self.std_=StandardScaler().fit(X)
        F=self._expand(self.std_.transform(X))
        labels=self._expand(X,names=True)
        y0=y.astype(float)-np.mean(y)
        f0=F-F.mean(axis=0)
        num=np.abs(f0.T@y0)
        denom=np.sqrt((f0**2).sum(axis=0)*(y0**2).sum())
        corr=np.divide(num,denom,out=np.zeros_like(num),where=denom>1e-10)
        self.feature_index_=int(np.argmax(corr))
        self.feature_name_=labels[self.feature_index_]
        self.logit_=LogisticRegression(C=self.C,max_iter=2000).fit(F[:,[self.feature_index_]],y)
        return self
    def predict_proba(self,X):
        F=self._expand(self.std_.transform(np.asarray(X,dtype=float)))
        return self.logit_.predict_proba(F[:,[self.feature_index_]])
    def predict(self,X):
        return self.classes_[np.argmax(self.predict_proba(X),axis=1)]


def basic_models():
    return OrderedDict([
        ('Regressão logística',make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=2000,random_state=42))),
        ('KNN (k=5)',make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5,weights='distance'))),
        ('Random Forest',RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=2,random_state=42,n_jobs=1)),
        ('SISSO-inspirado (1 termo)',SymbolicOneTermClassifier(C=1.0)),
    ])


def cv_predictions(X,y,splitter,groups=None,model=None):
    result=np.full(len(y),-1,dtype=int)
    for train,test in splitter.split(X,y,groups):
        trained=clone(model).fit(X[train],y[train])
        result[test]=trained.predict(X[test])
    assert (result>=0).all(), 'Há observações sem predição'
    return result
