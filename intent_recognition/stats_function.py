#This script contains the class of statistics functions that are used to evaluate the trained models


from math import sqrt

epsilon=1e-7 #Used to avoid division by 0

class StatsFunction: 
    #The class of statistics functions
    #Each object has a name to identify it, a function to calculate the statistic and a boolean to identify whether a lower value is better
    #Arithmetic operations are defined to aid in defining new stats functions

    def __init__(self,name,stat_function,minimise=True):
        self.name=name
        self.fn=stat_function
        self.minimise=True

    def __call__(self,label_counts):
        return self.fn(label_counts)
    
    def __add__(self,other):
        a=other
        if not isinstance(a,StatsFunction):
            a=StatsFunction.constant(a)
        def temp(x):
            return self(x)+a(x)
        return StatsFunction(self.name+' + '+a.name,temp)
    
    def __sub__(self,other):
        a=other
        if not isinstance(a,StatsFunction):
            a=StatsFunction.constant(a)
        def temp(x):
            return self(x)-a(x)
        return StatsFunction(self.name+' - '+a.name,temp)
    
    def __mul__(self,other):
        a=other
        if not isinstance(a,StatsFunction):
            a=StatsFunction.constant(a)
        def temp(x):
            return self(x)*a(x)
        return StatsFunction(self.name+'*'+a.name,temp)
    
    def __truediv__(self,other):
        a=other
        if not isinstance(a,StatsFunction):
            a=StatsFunction.constant(a)
        def temp(x):
            if a(x)==0:
                return self(x)/epsilon
            else:
                return self(x)/a(x)
        return StatsFunction(self.name+'/'+a.name,temp)
    
    def __rtruediv__(self,other):
        a=other
        if not isinstance(a,StatsFunction):
            a=StatsFunction.constant(a)
        def temp(x):
            if self(x)==0:
                return a(x)/epsilon
            else:
                return a(x)/self(x)
        return StatsFunction(a.name+'/'+self.name,temp)
    
    def __radd__(self,other):
        return self+other
    
    def __rmul__(self,other):
        return self*other
    
    def __rsub__(self,other):
        return (-1)*self+other


    @classmethod
    def constant(cls,x):
        def temp(_):
            return x
        return StatsFunction(str(x),temp)

    @classmethod
    def convexCombination(cls,fn1,fn2,p):
        return p*fn1+(1-p)*fn2

def error(label_counts):
    n=0
    m=0
    for target,prediction,count in label_counts:
        n+=count
        if target!=prediction:
            m+=count
    if n==0:
        n+=epsilon
    return m/n
    
error_rate=StatsFunction('error_rate',error)

def false_positive(label_counts):
    n=0
    m=0
    for target,prediction,count in label_counts:
        if target==0:
            n+=count
            if prediction==1:
                m+=count
    if n==0:
        n+=epsilon
    return m/n
    
false_positive_rate=StatsFunction('false_positive_rate',false_positive)

def false_negative(label_counts):
    n=0
    m=0
    for target,prediction,count in label_counts:
        if target==1:
            n+=count
            if prediction==0:
                m+=count
    if n==0:
        n+=epsilon
    return m/n
    
false_negative_rate=StatsFunction('false_negative_rate',false_negative)

def precision_fn(label_counts):
    n=0
    m=0
    for target,prediction,count in label_counts:
        if prediction==1:
            n+=count
            if target==1:
                m+=count
    if n==0:
        n+=epsilon
    return m/n
    
precision=StatsFunction('precision',precision_fn,False)

def recall_fn(label_counts):
    n=0
    m=0
    for target,prediction,count in label_counts:
        if target==1:
            n+=count
            if prediction==1:
                m+=count
    if n==0:
        n+=epsilon
    return m/n
    
recall=StatsFunction('recall',recall_fn,False) #AKA true positive rate

f_measure=StatsFunction('f-measure',2/((1/precision)+(1/recall)),False) #AKA dice coefficient or F1 score

def specificity_fn(label_counts):
    n=0
    m=0
    for target,prediction,count in label_counts:
        if target==0:
            n+=count
            if prediction==0:
                m+=count
    if n==0:
        n+=epsilon
    return m/n
    
specificity=StatsFunction('specificity',specificity_fn,False) #AKA true negative rate

stats=[error_rate,false_positive_rate,false_negative_rate,precision,recall,specificity,f_measure] #A list of all of the stats functions