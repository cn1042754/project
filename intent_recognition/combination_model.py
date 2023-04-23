#This script contains the models that combine the results from the phase models to predict the intent for the probaility version of the gesture models

import utils
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_check_sample_weight
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal

epsilon=1e-7 #Used to avoid division by 0 errors

#The class for all combination models
class CombinationModel:

    def __init__(self,name,model):
        self.name=name
        self.model=model

    def fit(self,x,y):
        self.model.fit(x,y)
    
    def __call__(self,preds):
        results=self.model.predict_proba(preds)
        results=np.array(results)[:,1]
        return results
    
    def __str__(self):
        return self.name
            
class BayesianModel(BaseEstimator,ClassifierMixin):
    #An sklearn style classifier that runs a bayesian model assuming that conditional on the class the results of the phase models are independent multivariate Gaussian distributions

    def fit(self,X,y):
        self.n_window_sizes=len(utils.window_sizes)
        X,y=check_X_y(X,y)
        self.n_phases=int(len(X[0])/self.n_window_sizes)
        self.classes_ = unique_labels(y)
        n_samples=len(y)
        pos_count=np.sum(y)
        total=len(X)
        pos_data=[X[i] for i in range(n_samples) if y[i]==1]
        neg_data=[X[i] for i in range(n_samples) if y[i]==0]
        pos_count=len(pos_data)
        neg_count=len(neg_data)
        self.pos_p=pos_count/total
        self.neg_p=1-self.pos_p
        self.pos_means=[]
        self.neg_means=[]
        self.pos_cov=[]
        self.neg_cov=[]
        for i in range(self.n_phases):
            pos_phase_data=np.array([datum[self.n_window_sizes*i:self.n_window_sizes*(i+1)] for datum in pos_data])
            neg_phase_data=np.array([datum[self.n_window_sizes*i:self.n_window_sizes*(i+1)] for datum in neg_data])
            self.pos_means.append(np.sum(pos_phase_data,0)/pos_count)
            self.neg_means.append(np.sum(neg_phase_data,0)/neg_count)
            self.pos_cov.append(np.cov(pos_phase_data,rowvar=False))
            self.neg_cov.append(np.cov(neg_phase_data,rowvar=False))
        self.pos_distributions=[multivariate_normal(self.pos_means[i],self.pos_cov[i]) for i in range(self.n_phases)]
        self.neg_distributions=[multivariate_normal(self.neg_means[i],self.neg_cov[i]) for i in range(self.n_phases)]
        return self

    def predict_proba(self,X):
        check_is_fitted(self)
        preds=check_array(X)
        phase_preds=np.split(preds,self.n_phases,1)
        pos_temp=[]
        neg_temp=[]
        for i in range(self.n_phases):
            pos_temp.append(self.pos_distributions[i].pdf(phase_preds[i]))
            neg_temp.append(self.neg_distributions[i].pdf(phase_preds[i]))
        pos_temp=np.prod(pos_temp,0)*self.pos_p
        neg_temp=np.prod(neg_temp,0)*self.neg_p
        results=np.stack([neg_temp/(pos_temp+neg_temp+epsilon),pos_temp/(pos_temp+neg_temp+epsilon)],1)
        return results
    
    def predict(self,X):
        probs=self.predict_proba(X)
        return np.argmax(probs,1)
    
#check_estimator(BayesianModel()) #This line was excluded due to unknown difficulties bringing the model in line with sklearn standards

base_models={
    'Bayesian Model': BayesianModel(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(100),
    'Logistic Regression': LogisticRegression()
}

class Combination:

    def __init__(self,name,model):
        self.model=CombinationModel(name,model)

    def __call__(self,n=6):
        return self.model
    
tree_combination=Combination('Decision Tree',DecisionTreeClassifier())
forest_combination=Combination('Random Forest',RandomForestClassifier(100))

#The list of base models
base_combinations=[Combination(name,base_models[name]) for name in base_models]

#Create a dictionary of Ada-boosted models
boosted_models={'Boosted Decision Stump':AdaBoostClassifier()}

#A list of boosted combination models
boosted_combinations=[]
for model_name in boosted_models:
    boosted_combinations.append(Combination(model_name,boosted_models[model_name]))

combinations=base_combinations+boosted_combinations #Compile a list of all combination models