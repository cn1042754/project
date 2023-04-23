#This script contains objects and subroutines used by the other scripts

import csv

#A list of the user ids and containers for the gesture and nongesture ids
user_ids=["user001","user002","user003","user004","user005","user006","user007","user008","user009","user010","user011","user012","user013","user014","user015","user016"]
gesture_ids={}
nongesture_ids={}

#Various lists of names and ids
sensors = ['Acc', 'Gyr', 'GRV', 'LAc'] #Names of the different sensors
phases=["reaching","alignment","withdrawal"] #Names of the gesture phases
window_sizes=[1,2] #Sizes of the windows used by the phase models
terminal_ids=["1","2","3","4","5","6","F"] #The ids of the different terminals
activity_ids=['W','B','I','M'] #The ids of the diferent activities performed to produce the non-gesture data (walking, on a bus or train, in a shop, mixed)

#index values for the data files
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds)
x_index = 3 #x-value
y_index = 4 #y-value
z_index = 5 #z-value
w_index = 6 #GRV w-value
e_index = 7 #Euclidean norms of unfiltered values

#index values for the times file
tu_index=0 #user id
tg_index=1 #gesture id
t1_index=2 #start of the reaching phase
t2_index=3 #transition from reaching to alignment
t3_index=4 #transition from alignment to withdrawal
t4_index=5 #end of the withdrawal phase

#Convert phase names to the indices for the start and end times for the phase in the times file
phase_time_index={
    "reaching":(t1_index,t2_index),
    "alignment":(t2_index,t3_index),
    "withdrawal":(t3_index,t4_index)
}

#Various file names
time_file_path="times.csv" #The file containing the phase transition times for each gesture
gesture_ids_file_path="gesture_ids.csv" #The file containing lists of gesture ids
nongesture_ids_file_path="nongesture_ids.csv" #The file containing lists of nongesture ids

#Load the gesture ids
with open(gesture_ids_file_path,'r') as file:
    data=csv.reader(file)
    for datum in data:
        gesture_ids[datum[0]]=datum[1:]

#Load the nongesture ids
with open(nongesture_ids_file_path,'r') as file:
    data=csv.reader(file)
    for datum in data:
        nongesture_ids[datum[0]]=datum[1:] 

#Fill in the entries for users that have no gestures or no nongestures
for uid in user_ids:
    if uid not in gesture_ids:
        gesture_ids[uid]=[]
    if uid not in nongesture_ids:
        nongesture_ids[uid]=[]  

def getDataFilePath(user_id,gesture=True,is_filtered=True):
    #generates the file path for the data from the user id, whether or not the data is filitered, and whether its a gesture file or a nongesture file
    if is_filtered:
        temp="data/filtered/"+user_id
    else:
        temp="data/original/"+user_id
    if gesture:
        temp+="-gestures.csv"
    else:
        temp+="-nongestures.csv"
    return temp

def getTrainingDataFilePath(phase,window_size,user_id,positive=True):
    #Gets the file path for a file containg training data for a phase model based on the phase name, window size, user id and whether or not it is the positive window file
    temp="data/for_training/"+phase+'-'+window_size+'s/'
    if positive:
        temp+="positive/"
    else:
        temp+="negative/"
    temp+=user_id+".csv"
    return temp

def convert(datum):
    #converts entries of a list to floats where possible
    temp=[]
    for x in datum:
        try:
            temp.append(float(x))
        except ValueError:
            temp.append(x)
    return temp

def getGestureData(user_id,gesture_id,gesture=True,is_filtered=True,to_convert=True):
    #Get the data from the specified gesture returning None if there is an error
    file_path=getDataFilePath(user_id,gesture,is_filtered)
    try:
        temp=[]
        with open(file_path,'r') as file:
            data=csv.reader(file)
            seen=False
            for datum in data:
                if datum[g_index]==gesture_id:
                    seen=True
                    if not to_convert:
                        temp.append(datum)
                    else:
                        temp.append(convert(datum))
                elif seen:
                    break
        return temp
    except Exception:
        return None

def getTimes(user_id,gesture_id):
    #Get the time data for the specified gesture
    try:
        with open(time_file_path,"r") as file:
            data=list(csv.reader(file))
    except Exception:
        return None
    for datum in data:
        if (datum[tu_index]==user_id) and (datum[tg_index]==gesture_id):
            return datum
    return None

def read(path,to_convert=False):
    #Read the csv data from the specified file performing float conversion if to_convert is True
    try:
        with open(path,'r') as f:
            if not to_convert:
                data=[x for x in csv.reader(f) if len(x)>0]
            else:
                data=[convert(x) for x in csv.reader(f) if len(x)>0]
        return data
    except Exception:
        return []


def getTrainingData(model_name):
    #Gets all training data for a specified model (the name is of the form <phase id>-<window size>s)
    #Returns the lists of the positive and negative window ids, the list of data and the list of target labels
    path='data/for-training/'+model_name+'/'
    temp_pos_ids=read(path+'positive_ids.csv')
    temp_neg_ids=read(path+'negative_ids.csv')
    pos_ids={}
    neg_ids={}
    for row in temp_pos_ids:
        uid=row[0]
        pos_ids[uid]=[wid for wid in row[1:]]
    for row in temp_neg_ids:
        uid=row[0]
        neg_ids[uid]=[wid for wid in row[1:]]
    positive_data=[]
    negative_data=[]
    for uid in pos_ids:
        data=read(path+'positive/'+uid+'.csv')
        for datum in data:
            if datum[0] in pos_ids[uid]:
                positive_data.append([float(x) for x in datum[1:]])
    for uid in neg_ids:
        data=read(path+'negative/'+uid+'.csv')
        for datum in data:
            if datum[0] in neg_ids[uid]:
                negative_data.append([float(x) for x in datum[1:]])
    data=positive_data+negative_data
    labels=[1]*len(positive_data)+[0]*len(negative_data)
    return (pos_ids,neg_ids),data,labels

def filterTrainingData(to_filter,mask):
    #Filters the given data based on the given mask returning the filtered list of data and the corresponding target labels
    gids,data,labels=to_filter
    pos_ids,neg_ids=gids
    i=0
    filtered_data=[]
    filtered_labels=[]
    for uid in pos_ids:
        for wid in pos_ids[uid]:
            if '-'.join(wid.split('-')[:-1]) in mask[uid]:
                filtered_data.append(data[i])
                filtered_labels.append(labels[i])
            i+=1
    for uid in neg_ids:
        for wid in neg_ids[uid]:
            if '-'.join(wid.split('-')[:-1]) in mask[uid]:
                filtered_data.append(data[i])
                filtered_labels.append(labels[i])
            i+=1
    return filtered_data,filtered_labels

def getData(uids=user_ids,time_before_zero=False):
    #Gets all gesture and nongesture data for the specified user ids filtering out data from after the contact time if specified
    #Returns a dictionary with the data and the a dictioary with the labels
    data={}
    labels={}
    gesture_uids=[uid for uid in gesture_ids if uid in uids]
    nongesture_uids=[uid for uid in nongesture_ids if uid in uids]
    for uid in gesture_uids:
        data[uid]={}
        labels[uid]={}
        for gid in gesture_ids[uid]:
            data[uid][gid]=[]
            labels[uid][gid]=1
    for uid in nongesture_uids:
        if uid not in data:
            data[uid]={}
            labels[uid]={}
        for gid in nongesture_ids[uid]:
            data[uid][gid]=[]
            labels[uid][gid]=0
    for uid in uids:
        path1=getDataFilePath(uid)
        path2=getDataFilePath(uid,False)
        gesture_data=read(path1,to_convert=True)
        nongesture_data=read(path2,to_convert=True)
        for datum in gesture_data+nongesture_data:
            try:
                gid=datum[g_index]
                if not time_before_zero or (isinstance(datum[t_index],float) and datum[t_index]<=0):
                    data[uid][gid].append(datum)
            except KeyError:
                pass
    return data,labels