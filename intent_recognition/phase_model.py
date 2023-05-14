#The module contains the phase model class and instantiates a range of phase models

import mask_generator,utils
from feature_extractor import extract_features
from sklearn.ensemble import RandomForestClassifier
import joblib
import os,math
import numpy as np
from copy import deepcopy as copy
from aggregation_function import max_aggregation

n_repetitions=5

def is_singular(x):#Determines if a list contains a single gesture or a list of them
	y=x
	count=0
	while True:
		if not isinstance(y,list):
			break
		else:
			if len(y)==0:
				break
			else:
				y=y[0]
				count+=1
	if count==3:
		return False
	elif count==2:
		return True
	else:
		return None
    

def get_interval(data,start,end): #Gets all data from the given gesture lying in a certian interval
    return [datum for datum in data if (datum[utils.t_index]>=start) and (datum[utils.t_index]<=end)]

class PhaseModel:

    def __init__(self,name,window_size,training_data,training_labels,random_state=None,extract_features=extract_features):
	#Initialise some variables
        self.extract_features=extract_features
        self.name=name+'-'+str(random_state)
        self.window_size=window_size
	#Attempt to load the phase model
        load_success=False
        if os.path.exists("phase_models/"+self.name+'.phase_model'):
            try:
                with open("phase_models/"+self.name+'.phase_model','rb') as file:
                    self.model=joblib.load(file)
                load_success=True
            except Exception:
                print("loading failed")
	#If the loading failed then create, train and store a new random forest
        if not load_success:
            self.model=RandomForestClassifier(n_estimators=100,random_state=random_state)
            self.model.fit(training_data,training_labels)
            with open("phase_models/"+self.name+'.phase_model','wb') as file:
                joblib.dump(self.model,file,compress=True)

    def __call__(self, data,aggregate=max_aggregation): #Run the phase model on the input gesture(s)
        singular=is_singular(data)
        if singular is None:
            return None
        elif singular:
            data=[data]
        result=[]
        for datum in data:
            start=float(datum[0][utils.t_index])-0.05
            end=float(datum[-1][utils.t_index])+0.05
            step_size=0.5*self.window_size
            temp=[start+i*step_size for i in range(math.ceil((end-start)/step_size)) if start+(i+2)*step_size<end]
            temp.append(end-self.window_size)
            datum_features=[self.extract_features.get_features(get_interval(datum,start,start+self.window_size)) for start in temp]
            temp=self.model.predict_proba(datum_features)
            temp=[a[1] for a in temp]
            result.append(aggregate(temp))
        if singular:
            result=result[0]
        return result

    def predict(self,user_id,gesture_id,gesture=True,is_filtered=True,aggregate=max_aggregation):#Same as __call__ but loads a specified gesture
        self.extract_features.load_gesture(user_id,gesture_id,gesture,is_filtered)
        start=float(self.extract_features.current_gesture_data[0][utils.t_index])
        end=float(self.extract_features.current_gesture_data[-1][utils.t_index])
        step_size=0.5*self.window_size
        temp=[start+i*step_size for i in range(math.ceil((end-start)/step_size)) if start+(i+2)*step_size<end]
        temp.append(end-self.window_size)
        data=[self.extract_features(user_id,gesture_id,start,start+self.window_size,gesture,is_filtered) for start in temp]
        temp=self.model.predict_proba(data)
        temp=[a[1] for a in temp]
        return aggregate(temp)
    
    def getTopKFeatures(self,k=5): #Returns the feature indices for the top k features for this model by gini importance
        importances=self.model.feature_importances_
        temp=[(i,importances[i]) for i in range(len(importances))]
        temp.sort(reverse=True,key=(lambda x: x[1]))
        temp=[i for i,_ in temp]
        del temp[k:]
        return temp

    def combine(self,other_phase_models=[],new_name=None): #Create a new phase model that is a copy of this one combined with a list of others
        temp=copy(self)
        if new_name is not None:
            temp.name=new_name
        for model in other_phase_models:
            temp.model.estimators_+=model.model.estimators_
            temp.model.n_estimators=len(temp.model.estimators_)
        return temp

def combine_phase_models(models=[],name=None): #Combine a list of phase models
    if len(models)==0:
        return None
    else:
        return models[0].combine(models[1:],name)

def create_models(phase,window_size,mask_gen,num_splits=mask_generator.default_num_splits,random_states=[0]): #Create all phase models for a given phase, window size amd mask generator
    data=utils.getTrainingData(phase+'-'+str(window_size)+'s')
    if not mask_gen.test_indexing or mask_gen.user or mask_gen.terminal:
        test_split_indices=[0]
        val_split_indices=list(range(num_splits-1))
    else:
        test_split_indices=list(range(num_splits))
        val_split_indices=list(range(num_splits-1))
    for test_split_index in test_split_indices:
        for val_split_index in val_split_indices:
            training_data,training_labels=utils.filterTrainingData(data,mask_gen.get_train(val_split_index,test_split_index))
            name='-'.join([mask_gen.name,phase,str(window_size)+'s',str(test_split_index),str(val_split_index)])
            for random_state in random_states:
                if not os.path.exists("phase_models/"+name+'-'+str(random_state)+'.phase_model'):
                    model=PhaseModel(name,window_size,training_data,training_labels,random_state)
                    print(model.name)

def create_limited_sensor_models(phase,window_size,mask_gen,num_splits=mask_generator.default_num_splits,random_states=[0]):#Create all phase models for a given phase, window size amd mask generator using limited sensors
    data=utils.getTrainingData('limited_sensors_'+phase+'-'+str(window_size)+'s')
    if not mask_gen.test_indexing:
        test_split_indices=[-1]
        val_split_indices=list(range(num_splits-1))
    elif mask_gen.user or mask_gen.terminal:
        test_split_indices=[0]
        val_split_indices=list(range(num_splits-1))
    else:
        test_split_indices=list(range(num_splits))
        val_split_indices=list(range(num_splits-1))
    for test_split_index in test_split_indices:
        for val_split_index in val_split_indices:
            training_data,training_labels=utils.filterTrainingData(data,mask_gen.get_train(val_split_index,test_split_index))
            name='-'.join([mask_gen.name,'limited_sensors',phase,str(window_size)+'s',str(test_split_index),str(val_split_index)])
            for random_state in random_states:
                if not os.path.exists("phase_models/"+name+'-'+str(random_state)+'.phase_model'):
                    model=PhaseModel(name,window_size,training_data,training_labels,random_state)
                    print(model.name)

if __name__=="__main__":
    #Create the general phase models
    for phase in utils.phases:
        for window_size in utils.window_sizes:
            create_models(phase,window_size,mask_generator.general_mask)

    #Create the phase models to evaluate the method on an unseen user
    for uid in mask_generator.user_masks:
        for phase in utils.phases:
            for window_size in utils.window_sizes:
                create_models(phase,window_size,mask_generator.user_masks[uid],random_states=list(range(mask_generator.default_num_splits)))

    #Create the phase models to evaluate the method on an unseen terminal
    for tid in mask_generator.terminal_masks:
        for phase in utils.phases:
            for window_size in utils.window_sizes:
                create_models(phase,window_size,mask_generator.terminal_masks[tid],random_states=list(range(mask_generator.default_num_splits)))

    #Create the phase models that use limited sensors
    for phase in utils.phases:
        for window_size in utils.window_sizes:
            create_limited_sensor_models(phase,window_size,mask_generator.general_mask)

