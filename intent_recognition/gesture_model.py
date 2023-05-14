#This module contains the main gesture model class and performs the experiments

import mask_generator,utils
import os,joblib,time
import numpy as np
from phase_model import PhaseModel,combine_phase_models
from concurrent.futures import ProcessPoolExecutor as Executor
from aggregation_function import max_aggregation,mean_aggregation
from stats_function import StatsFunction,false_positive_rate,false_negative_rate,error_rate,stats
import matplotlib.pyplot as plt
from feature_extractor import extract_features,extract_limited_features
from combination_model import combinations,tree_combination,forest_combination

threshold_step_size=0.001 #The gap between candidate decision thresholds
chunk_size=1000 #The number of gestures to pass to the phase models at one time

def time_str(): #Returns a string of the current time
    return time.strftime("%H:%M:%S",time.localtime())

def topKIndices(l,k=5,reverse=True): #Returns the indices of the largest k values in the given list (or smallest when reverse is false)
    temp=[(i,l[i]) for i in range(len(l))]
    temp.sort(reverse=reverse,key=(lambda x: x[1]))
    return [i for i,_ in temp[:k]]

class Predict: #A class that manages the combination models

    def __init__(self,gesture_model,combination_models={},threshold=0):
        self.models=combination_models
        self.threshold=threshold
        self.gesture_model=gesture_model

    def __call__(self,name,data=None): #Return the class predictions of the specified model on the given data or the test data corresponding to the model
        model=self.models[name]
        if data is None:
            results=self.predictions[name]
        else:
            results=model(data)
        return [result>=self.threshold for result in results]
    
    def set_threshold(self,threshold): #Set the decision threshold to use
        self.threshold=threshold

    def set_models(self,combination_models): #Change the type of combination model  to use
        self.models=combination_models
        try:
            with open(self.gesture_model.path+'combination-'+self.models['0-0'].name+'-'+'-'.join(self.gesture_model.phases)+'.predictions','rb') as f:
                self.predictions=joblib.load(f)
        except Exception:
            self.predictions={}
            for name in self.models:
                model=self.models[name]
                data=[pred for pred,_ in self.gesture_model.test_results[name]]
                preds=model(data)
                self.predictions[name]=preds
            with open(self.gesture_model.path+'combination-'+self.models['0-0'].name+'-'+'-'.join(self.gesture_model.phases)+'.predictions','wb') as f:
                joblib.dump(self.predictions,f,True)

class GestureModel:

    def __init__(self,name,mask_gen,val_indices=list(range(mask_generator.default_num_splits-1)),test_indices=list(range(mask_generator.default_num_splits)),phases=utils.phases,window_sizes=utils.window_sizes,aggregate=max_aggregation,notes='',time_before_zero=False,removed=[],loading_updates=False,limited_sensors=False,combination_model=None):
        #Initialise some variables
        self.removed=removed
        self.loading_updates=loading_updates
        self.notes=notes
        if time_before_zero:
            self.notes+='Only data from before the contact time are used\n'
        self.optimised=False
        self.best_threshold=0
        self.thresholds=[threshold_step_size*i for i in range(1,int(1/threshold_step_size))]
        self.name=name+'-'+mask_gen.name
        self.name_no_mask=name
        self.mask_gen=mask_gen
        path='gesture_models/'+self.name+'/'
        self.path=path
        try:
            os.mkdir(path)
        except:
            pass
        self.val_indices=val_indices
        self.aggregate=aggregate
        test_indexing=mask_gen.test_indexing
        if (test_indexing and not mask_gen.user and not mask_gen.terminal):
            self.test_indices=test_indices
            self.random_states=[0]
        else:
            self.test_indices=[0]
            self.random_states=test_indices
        self.phases=phases
        self.extract_features=extract_features
        if limited_sensors:
            self.phases=['limited_sensors-'+phase for phase in phases]
            self.extract_features=extract_limited_features
        self.limited_sensors=limited_sensors
        self.window_sizes=window_sizes
        self.val_results={}
        self.combined_phase_models={}
        exists=True
        val_results_to_calc=[]
        if loading_updates:
            print("Attempting to load validation results",time_str())
        for uid in utils.user_ids:
            for test_index in self.test_indices:
                for val_index in self.val_indices:
                    for random_state in self.random_states:
                        try:
                            name=str(test_index)+'-'+str(val_index)+'-'+str(random_state)
                            with open(path+uid+'-'+name+'.results','rb') as f:
                                temp=joblib.load(f)
                                if name in self.val_results:
                                    self.val_results[name]+=temp
                                else:
                                    self.val_results[name]=temp
                        except:
                            exists=False
                            val_results_to_calc.append((uid,test_index,val_index,random_state))
        if len(removed)>0:
            temp=self.val_results
            self.val_results={}
            for test_index in self.test_indices:
                for val_index in self.val_indices:
                    for random_state in self.random_states:
                        name=str(test_index)+'-'+str(val_index)+'-'+str(random_state)
                        self.val_results[name]=[]
                        for result in temp[name]:
                            predictions,target=result
                            new_predictions=[]
                            for i in range(len(self.phases)+len(removed)):
                                if i not in removed:
                                    new_predictions+=predictions[i*len(self.window_sizes):(i+1)*len(self.window_sizes)]
                            new_result=(new_predictions,target)
                            self.val_results[name].append(new_result)
        if loading_updates:
            if len(val_results_to_calc)==0:
                print('Successful')
            else:
                print('Failed, will calculate validation results for',len(val_results_to_calc),'validation splits')
        self.test_results={}
        test_results_to_calc=[]
        if loading_updates:
            print('Attempting to load test results',time_str())
        for uid in utils.user_ids:
            for test_index in self.test_indices:
                for random_state in self.random_states:
                    try:
                        name=str(test_index)+'-'+str(random_state)
                        with open(path+uid+'-'+name+'.results','rb') as f:
                            temp=joblib.load(f)
                            if name in self.test_results:
                                self.test_results[name]+=temp
                            else:
                                self.test_results[name]=temp
                    except:
                        exists=False
                        test_results_to_calc.append((uid,test_index,random_state))
        if len(removed)>0:
            temp=self.test_results
            self.test_results={}
            for test_index in self.test_indices:
                for random_state in self.random_states:
                    name=str(test_index)+'-'+str(random_state)
                    self.test_results[name]=[]
                    for result in temp[name]:
                        predictions,target=result
                        new_predictions=[]
                        for i in range(len(self.phases)+len(removed)):
                            if i not in removed:
                                new_predictions+=predictions[i*len(self.window_sizes):(i+1)*len(self.window_sizes)]
                        new_result=(new_predictions,target)
                        self.test_results[name].append(new_result)
        if loading_updates:
            if len(test_results_to_calc)==0:
                print('Successful')
            else:
                print('Failed, will produce and calculate the combined phase models and test results for',len(test_results_to_calc),'test splits')
        if loading_updates:
            print('Loading combined phase models',time_str())
        for test_index in self.test_indices:
            for random_state in self.random_states:
                try:
                    name=str(test_index)+'-'+str(random_state)
                    with open(path+name+'.phase_models','rb') as f:
                        self.combined_phase_models[name]=joblib.load(f)
                except:
                    combined_phase_models=[]
                    for phase in self.phases:
                        for window_size in window_sizes:
                            phase_model_names=[('-'.join([mask_gen.name,phase,str(window_size)+'s',str(test_index),str(val_index)]),window_size) for val_index in val_indices]
                            phase_models=[PhaseModel(phase_model_name,window_size,[],[],random_state,extract_features=self.extract_features) for phase_model_name,window_size in phase_model_names]
                            combined_phase_models.append(combine_phase_models(phase_models,name+'-'+phase+'-'+str(window_size)))
                    self.combined_phase_models[name]=combined_phase_models
                    with open(path+name+'.phase_models','wb') as f:
                        joblib.dump(combined_phase_models,f,True)  
        #If at least one load failed then produce the missing data
        if not exists:
            data={}
            for uid,test_index,val_index,random_state in val_results_to_calc:
                if uid not in data:
                    print(uid,'Loading Data',time_str())
                    data,labels=utils.getData([uid],time_before_zero)
                val_mask=mask_gen.get_validation(val_index,test_index)
                val_data=[]
                val_labels=[]
                if uid in val_mask:
                    for gid in val_mask[uid]:
                        try:
                            val_data.append(data[uid][gid])
                            val_labels.append(labels[uid][gid])
                        except KeyError:
                            pass
                n=len(val_labels)
                name=str(test_index)+'-'+str(val_index)+'-'+str(random_state)
                print(uid,name,'Loading phase models',time_str())
                phase_model_names=[('-'.join([mask_gen.name,phase,str(window_size)+'s',str(test_index),str(val_index)]),window_size) for phase in self.phases for window_size in self.window_sizes]
                phase_models=[PhaseModel(phase_model_name,window_size,[],[],random_state,extract_features=self.extract_features) for phase_model_name,window_size in phase_model_names]
                print(uid,name,'Running Phase Models on validation data',time_str())
                val_results=[]
                if n>0:
                    with Executor(len(phase_models)) as executor:
                        model_runs={}
                        val_data_split=[val_data[i:i+chunk_size] for i in range(0,len(val_data),chunk_size)]
                        for split in range(len(val_data_split)):
                            for model in phase_models:
                                model_runs[model.name]=executor.submit(model,val_data_split[split],aggregate=self.aggregate)
                            temp=[model_runs[model.name].result() for model in phase_models]
                            val_results+=[[temp[j][i] for j in range(len(phase_models))] for i in range(len(val_data_split[split]))]
                print(uid,name,"Counting Phase Model Results",time_str())
                results=[(val_results[i],val_labels[i]) for i in range(len(val_results))]
                if name in self.val_results:
                    self.val_results[name]+=results
                else:
                    self.val_results[name]=results
                print(uid,name,"Storing calculated results",time_str())
                with open(path+uid+'-'+name+'.results','wb') as f:
                    joblib.dump(results,f,True)

            for uid,test_index,random_state in test_results_to_calc:
                if uid not in data:
                    print(uid,'Loading Data',time_str())
                    data,labels=utils.getData([uid],time_before_zero)
                test_mask=mask_gen.get_testing(test_index)
                test_data=[]
                test_labels=[]
                if uid in test_mask:
                    for gid in test_mask[uid]:
                        try:
                            test_data.append(data[uid][gid])
                            test_labels.append(labels[uid][gid])
                        except KeyError:
                            pass
                n=len(test_labels)
                name=str(test_index)+'-'+str(random_state)
                print(uid,name,"Loading Phase Models",time_str())
                if not name in self.combined_phase_models:
                    combined_phase_models=[]
                    for phase in self.phases:
                        for window_size in window_sizes:
                            phase_model_names=[('-'.join([mask_gen.name,phase,str(window_size)+'s',str(test_index),str(val_index)]),window_size) for val_index in val_indices]
                            phase_models=[PhaseModel(phase_model_name,window_size,[],[],random_state,extract_features=self.extract_features) for phase_model_name,window_size in phase_model_names]
                            combined_phase_models.append(combine_phase_models(phase_models,name+'-'+phase+'-'+str(window_size)))
                    self.combined_phase_models[name]=combined_phase_models
                    with open(path+name+'.phase_models','wb') as f:
                        joblib.dump(combined_phase_models,f,True)
                else:
                    combined_phase_models=self.combined_phase_models[name]
                print(uid,name,"Running Phase Models on test data",time_str())
                test_results=[]
                if n>0:
                    with Executor(len(combined_phase_models)) as executor:
                        model_runs={}
                        test_data_split=[test_data[i:i+chunk_size] for i in range(0,len(test_data),chunk_size)]
                        for split in range(len(test_data_split)):
                            for model in combined_phase_models:
                                model_runs[model.name]=executor.submit(model,test_data_split[split],aggregate=self.aggregate)
                            temp=[model_runs[model.name].result() for model in combined_phase_models]
                            test_results+=[[temp[j][i] for j in range(len(combined_phase_models))] for i in range(len(test_data_split[split]))]
                print(uid,name,"Counting Results",time_str())
                results=[(test_results[i],test_labels[i]) for i in range(len(test_results))]
                if name in self.test_results:
                    self.test_results[name]+=results
                else:
                    self.test_results[name]=results
                print(uid,name,'Storing calculated results',time_str())
                with open(path+uid+'-'+name+'.results','wb') as f:
                    joblib.dump(results,f,True)
        
        self.set_combination_model(combination_model)


    def getTestIDs(self,name): #Produces a list of the test ids
        test_index=int(name.split('-')[0])
        test_mask=self.mask_gen.get_testing(test_index)
        temp=[]
        for uid in test_mask:
            temp+=[(uid,gid) for gid in test_mask[uid]]
        return temp
        
    def evaluate(self,stats,threshold=None,filter=(lambda x: True)): #Evaluate the gesture model on each of the given statistics for the test data
        results={}
        if threshold is None:
            self.predict.set_threshold(self.best_threshold)
        else:
            self.predict.set_threshold(threshold)
        for name in [stat.name for stat in stats]:
            results[name]=[]
        for name in self.test_results:
            targets=[target for _,target in self.test_results[name]]
            preds=self.predict(name)
            counts=[[0,0],[0,0]]
            testIDs=self.getTestIDs(name)
            for i in range(len(preds)):
                if filter(testIDs[i]):
                    counts[int(targets[i])][int(preds[i])]+=1
            labels=[(i,j,counts[i][j]) for i in range(2) for j in range(2)]
            for stat_fn in stats:
                results[stat_fn.name].append(stat_fn(labels))
        for name in results:
            results[name]=mean_aggregation(results[name])
        return results
    
    def getEER(self,graph=False): #Optimises the model for the EER and returns this value, grpahing the FPR, FNR and error when graph is True
        diff_stat=StatsFunction('diff',(lambda x: abs(false_negative_rate(x)-false_positive_rate(x))))
        eer=StatsFunction('eer',0.5*false_negative_rate+0.5*false_positive_rate)
        results={}
        for threshold in self.thresholds:
            results[threshold]=self.evaluate([diff_stat,eer,false_negative_rate,false_positive_rate,error_rate],threshold)
        best_threshold=self.thresholds[0]
        best_diff=results[best_threshold]['diff']
        for threshold in self.thresholds:
            diff=results[threshold]['diff']
            if diff<best_diff:
                best_threshold=threshold
                best_diff=diff
        if graph:
            fns=[results[threshold][false_negative_rate.name] for threshold in self.thresholds]
            fps=[results[threshold][false_positive_rate.name] for threshold in self.thresholds]
            errors=[results[threshold][error_rate.name] for threshold in self.thresholds]
            plt.plot(self.thresholds,fns,label='False Negative Rate')
            plt.plot(self.thresholds,fps,label='False Positive Rate')
            plt.plot(self.thresholds,errors,label='Error Rate')
            plt.xlabel('Threshold')
            plt.legend()
            plt.show()
        self.best_threshold=best_threshold
        return results[best_threshold]['eer']
    
    def remove_phases(self,phases): #Return a new gesture model with the specified phases removed
        old_phases=[]
        for i in range(len(self.phases)+len(self.removed)):
            if i in self.removed:
                old_phases.append('')
            else:
                old_phases.append(self.phases[i])
        new_phases=[temp_phase for temp_phase in self.phases if temp_phase not in phases]
        removed=[]
        for i in range(len(old_phases)):
            if old_phases[i] in phases:
                removed.append(i)
        return GestureModel(self.name_no_mask,self.mask_gen,self.val_indices,self.test_indices,new_phases,self.window_sizes,self.aggregate,notes=self.notes+','.join(phases)+' removed\n',removed=self.removed+removed,loading_updates=self.loading_updates,combination_model=self.combination_model)

    def set_combination_model(self,combination_model=None): #Set and train the combination models
        self.combination_model=combination_model
        self.predict=Predict(self)
        if combination_model is None:
            combination_model=combinations[0]
        try:
            with open(self.path+'combination-'+combination_model().name+'-'+'-'.join(self.phases)+'.combination_models','rb') as f:
                combination_models=joblib.load(f)
                self.predict.set_models(combination_models)
        except Exception:
            combination_models={}
            for test_index in self.test_indices:
                for random_state in self.random_states:
                    name1=str(test_index)+'-'+str(random_state)
                    combination_models[name1]=combination_model(len(self.phases)*len(self.window_sizes))
                    temp=[]
                    for val_index in self.val_indices:
                        name2=str(test_index)+'-'+str(val_index)+'-'+str(random_state)
                        temp+=self.val_results[name2]
                    x=[pred for pred,_ in temp]
                    y=[target for _,target in temp]
                    combination_models[name1].fit(x,y)
            self.predict.set_models(combination_models)
            with open(self.path+'combination-'+combination_model().name+'-'+'-'.join(self.phases)+'.combination_models','wb') as f:
                joblib.dump(combination_models,f,True)

    def getFeatureImportance(self,phases=None,k=0,prefix=''): #Return the top k features by gini importance filtering by phase
        if phases is None:
            phases=self.phases
        feature_names=self.extract_features.feature_names
        importances={}
        for name in feature_names:
            importances[name]=0
        for test_index in self.test_indices:
            for val_index in self.val_indices:
                for random_state in self.random_states:
                    phase_model_names=[('-'.join([self.mask_gen.name,phase,str(window_size)+'s',str(test_index),str(val_index)]),window_size) for phase in phases for window_size in self.window_sizes]
                    phase_models=[PhaseModel(phase_model_name,window_size,[],[],random_state,extract_features=self.extract_features) for phase_model_name,window_size in phase_model_names]
                    for model in phase_models:
                        temp=model.model.feature_importances_
                        temp=topKIndices(temp)
                        for i in temp:
                            importances[feature_names[i]]+=1
        importances=[(name,importances[name]) for name in feature_names]
        importances.sort(reverse=True,key=(lambda x: x[1]))
        if k>0:
            del importances[k:]
        for name,importance in importances:
            print(prefix,name,importance)
        return importances
    
    def getPhaseImportance(self,comb='forest',average_over_window_sizes=True,prefix=''): #Get the gini importances for the phases
        if comb=='tree':
            self.set_combination_model(tree_combination)
        else:
            self.set_combination_model(forest_combination)
        combination_models=self.predict.models
        phases=self.phases
        window_sizes=self.window_sizes
        importances={}
        for phase in phases:
            importances[phase]=[0 for _ in range(len(window_sizes))]
        count=0
        for name in combination_models:
            model=combination_models[name]
            temp=model.model.feature_importances_
            count+=1
            for i in range(len(phases)):
                phase=phases[i]
                for j in range(len(window_sizes)):
                    importances[phase][j]+=temp[i*len(window_sizes)+j]
        for phase in phases:
            for j in range(len(window_sizes)):
                importances[phase][j]/=count
        if average_over_window_sizes:
            importances=[(phase,np.mean(importances[phase])) for phase in phases]
            importances.sort(reverse=True,key=(lambda x: x[1]))
            for phase,importance in importances:
                print(prefix,phase,importance)
        else:
            importances=[(phase,window_sizes[j],importances[phase][j]) for phase in phases for j in range(len(window_sizes))]
            importances.sort(reverse=True,key=(lambda x: x[2]))
            for phase,window_size,importance in importances:
                print(prefix,phase,str(window_size)+'s',importance)
        return importances

def try_combinations(gesture_model,name='',combination_models=combinations,display=True,stats=stats): #Gets the EER and evaluates the given gesture model for each given combination model and statistic
    model_stats={}
    for model in combination_models:
        gesture_model.set_combination_model(model)
        eer=gesture_model.getEER()
        eval_stats=gesture_model.evaluate(stats)
        eval_stats['eer']=eer
        model_stats[model().name]=eval_stats
        if display:
            for stat in eval_stats:
                print(name,model().name,stat+':',eval_stats[stat])
            print()
    return model_stats
    
def average_stats(stats_list): #Average the statistics over a list
    temp={}
    counts={}
    for stats in stats_list:
        for model_name in stats:
            temp[model_name]={}
            counts[model_name]={}
            for stat in stats[model_name]:
                if stat in temp[model_name]:
                    temp[model_name][stat]+=stats[model_name][stat]
                    counts[model_name][stat]+=1
                else:
                    temp[model_name][stat]=stats[model_name][stat]
                    counts[model_name][stat]=1
    for model_name in temp:
        for stat in temp[model_name]:
            temp[model_name][stat]/=counts[model_name][stat]
    return temp
    
def print_stats(stats,prefix=''): #Print a list of stats with a given prefix
    for model_name in stats:
        for stat in stats[model_name]:
            print(prefix,model_name,stat+':',stats[model_name][stat])
        print()
    print()

def getPhaseModelNames(mask,phase,test_indexes=list(range(mask_generator.default_num_splits)),val_indices=list(range(mask_generator.default_num_splits-1))): #Returns a list of phase models
    if mask.user or mask.terminal:
        test_indices=[0]
        random_states=test_indexes
    else:
        test_indices=test_indexes
        random_states=[0]
    return [('-'.join([mask.name,phase,str(window_size)+'s',str(test_index),str(val_index)]),window_size,random_state) for val_index in val_indices for test_index in test_indices for window_size in utils.window_sizes for random_state in random_states]

def getFeatureImportances(phase_model_names,k=5): #Gets the feature importances for each phase model specified
    temp=[]
    for name,window_size,random_state in phase_model_names:
        model=PhaseModel(name,window_size,[],[],random_state)
        temp+=model.getTopKFeatures(k)
    return temp

def tallyFeatureImportances(feature_indices,feature_names,k=10,prefix=''): #Tallies and prints the top k features
    counts={}
    for index in feature_indices:
        if index in counts:
            counts[index]+=1
        else:
            counts[index]=1
    temp=[(feature_names[i],counts[i]) for i in counts]
    temp.sort(reverse=True,key=(lambda x: x[1]))
    del temp[k:]
    print(prefix+'Top',k,'Features:')
    for name,count in temp:
        print(name,count)
    print()

def printFeatureImportances(phases,masks,feature_names,k1=5,k2=10,prefix=''): #Retrieve and Print the feature importances
    indices=[]
    for phase in phases:
        temp=[]
        for mask in masks:
            temp+=getPhaseModelNames(mask,phase)
        temp=getFeatureImportances(temp,k=k1)
        tallyFeatureImportances(temp,feature_names,k=k2,prefix=prefix+phase+' ')
        indices+=temp
    tallyFeatureImportances(indices,feature_names,k=k2,prefix=prefix)
    print()

if __name__=="__main__":
    general_model=GestureModel('General',mask_generator.general_mask,combination_model=forest_combination)
    #Get the phase importances
    print("Phase Importances")
    _=general_model.getPhaseImportance()
    print()
    #Get the overall evaluation results
    try_combinations(general_model,'Overall')
    print()
    #For each phase evaluate the model with that phase removed
    for phase in utils.phases:
        temp_model=general_model.remove_phases([phase])
        try_combinations(temp_model,phase+' removed')
        print()
        del temp_model
    #For each phase evaluate the model using only that phase
    for phase in utils.phases:
        phases_to_remove=[phase_name for phase_name in utils.phases if phase_name!=phase]
        temp_model=general_model.remove_phases(phases_to_remove)
        try_combinations(temp_model,phase+' only')
        print()
        del temp_model
    del general_model
    #Evaluate the gesture model for in-store real-time usage
    general_model_no_withdrawal_before_zero_time=GestureModel('General No Withdrawal',mask_generator.general_mask,phases=[phase for phase in utils.phases if phase!='withdrawal'],time_before_zero=True)
    try_combinations(general_model_no_withdrawal_before_zero_time,'No withdrawal before 0 time')
    print()
    del general_model_no_withdrawal_before_zero_time
    #Evaluate the model on limited sensors
    limited_sensor_model=GestureModel('limited sensors',mask_generator.general_mask,limited_sensors=True)
    try_combinations(limited_sensor_model,'Limited Sensors')
    print()
    del limited_sensor_model
    #Evaluate the model for an unseen user
    user_stats=[]
    for uid in utils.user_ids:
        user_model=GestureModel('User Blind-'+uid,mask_generator.user_masks[uid],test_indices=[0])
        user_stats.append(try_combinations(user_model,'User '+uid,display=False))
        del user_model
    print_stats(average_stats(user_stats),'Unseen User')
    #Evaluate the model for an unseen terminal
    terminal_stats=[]
    for tid in utils.terminal_ids:
        terminal_model=GestureModel('Terminal Blind-'+tid,mask_generator.terminal_masks[tid],test_indices=[0])
        terminal_stats.append(try_combinations(terminal_model,'Terminal '+tid,display=False))
        del terminal_model
    print_stats(average_stats(terminal_stats),'Unseen Terminal')
    #Get the feature importances
    masks=[mask_generator.general_mask]+list(mask_generator.user_masks.values())+list(mask_generator.terminal_masks.values())
    phases=utils.phases
    printFeatureImportances(phases,masks,extract_features.feature_names,prefix='Overall ')
    phases=['limited_sensors-'+ phase for phase in phases]
    printFeatureImportances(phases,[mask_generator.general_mask],extract_limited_features.feature_names,prefix='Limited Sensors ')
