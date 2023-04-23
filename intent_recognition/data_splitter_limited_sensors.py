#Similar to the data splitter script this segments the data according to the phase transition times
#It takes a window of appropriate size at the beginning, middle and end of each phase and splits the nongestures into windows
#Finally it extracts the features from each of these windows and stores the results which are used to train phase models
#Unlike the data splitter script this script only extracts the features from the acclerometer and gyroscope data
#This is in order to analyse the architecture's usability with limited sensors

import utils,os,math,csv
from feature_extractor import extract_limited_features as extract_features

class NumFeaturesError(Exception):
     #A type of error when the number of features generated is incorrect

    def __init__(self,length):
        self.length=length
        self.target_length=len(extract_features.feature_names)
        self.message='Length of features incorrect: '+str(length)+' instead of '+str(self.target_length)
        super().__init__(self.message)

phases=["reaching","alignment","withdrawal"]
window_sizes=[1,2]

def getGestureWindows(user_id,gesture_id,phases=phases,window_sizes=window_sizes):
    #Gets the positive and negative phase windows from a particular gesture
    #The positive windows occur at the beginning, middle and end of the target phase
    #The negative windows are any windows that don't overlap the target phase
    extract_features.load_gesture(user_id,gesture_id)
    pos_temp={}
    neg_temp={}
    for phase in phases:
        for window_size in window_sizes:
            index=phase+'-'+str(window_size)
            start_time_index,end_time_index=utils.phase_time_index[phase]
            times=utils.getTimes(user_id,gesture_id)
            if times is not None:
                phase_start_time=float(times[start_time_index])
                phase_end_time=float(times[end_time_index])
                step=0.5*window_size
                if extract_features.get_interval(phase_start_time,phase_end_time) is not None:
                    phase_middle_time=(phase_start_time+phase_end_time)/2
                    temp=[(phase_start_time,phase_start_time+window_size),(phase_middle_time-step,phase_middle_time+step),(phase_end_time-window_size,phase_end_time)]
                    pos_temp[index]=[(user_id,gesture_id,start,end,True) for (start,end) in temp if extract_features.get_interval(start,end) is not None]
                data=extract_features.current_gesture_data
                start=float(data[0][utils.t_index])
                end=float(data[-1][utils.t_index])
                num_steps=math.ceil((end-start)/step)-1
                temp=[(start+step*i,start+step*i+window_size) for i in range(num_steps) if (start+step*i+window_size<=phase_start_time) or (start+step*i>=phase_end_time)]
                neg_temp[index]=[(user_id,gesture_id,start,end,True) for (start,end) in temp if extract_features.get_interval(start,end) is not None]
    return pos_temp,neg_temp


def getNonGestureWindows(user_id,gesture_id,phases=phases,window_sizes=window_sizes):
    #Splits a given non-gesture into windows overlapping by 50% to avoid missed details
    temp={}
    for phase in phases:
        for window_size in window_sizes:
            index=phase+'-'+str(window_size)
            step=0.5*window_size
            start=-4
            temp[index]=[]
            while start<-2:
                temp[index].append((user_id,gesture_id,start,start+window_size,False))
                start+=step
    return temp

print('Initializing\n') #Initialise the windows dictionaries

pos_windows={}
neg_windows={}
for phase in phases:
    for window_size in window_sizes:
        temp=phase+'-'+str(window_size)
        pos_windows[temp]={}
        neg_windows[temp]={}
        for uid in utils.user_ids:
            neg_windows[temp][uid]=[]
            pos_windows[temp][uid]=[]

print('Getting Window Intervals\n') #Get the windows for each gesture and non-gesture

for user_id in utils.user_ids:
    display_string='Getting Window Intervals\n'+user_id+'\nGesture'
    n=len(utils.gesture_ids[user_id])
    i=1
    for gesture_id in utils.gesture_ids[user_id]:
        print(display_string+' : '+str(i)+'/'+str(n))
        pos,neg=getGestureWindows(user_id,gesture_id)
        for index in pos:
            count=1
            for uid,gid,start,end,gesture in pos[index]:
                pos_windows[index][uid].append((gid,str(count),start,end,gesture))
                count+=1
        for index in neg:
            count=1
            for uid,gid,start,end,gesture in neg[index]:
                neg_windows[index][uid].append((gid,str(count),start,end,gesture))
                count+=1
        i+=1
    i=1
    n=len(utils.nongesture_ids[user_id])
    display_string='Getting Window Intervals\n'+user_id+'\nNonGesture'
    for gesture_id in utils.nongesture_ids[user_id]:
        print(display_string+' : '+str(i)+'/'+str(n))
        neg=getNonGestureWindows(user_id,gesture_id)
        for index in neg:
            count=1
            for uid,gid,start,end,gesture in neg[index]:
                neg_windows[index][uid].append((gid,str(count),start,end,gesture))
                count+=1
        i+=1

#Generate and store the window indices, window time intervals and the features for each window

for phase in phases:
    for window_size in window_sizes:
        index=phase+'-'+str(window_size)
        print()
        print(index,'\n')
        filepath="data/for-training/limited_sensors_"+index+'s/'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        if not os.path.exists(filepath+'positive/'):
            os.makedirs(filepath+'positive/')
        if not os.path.exists(filepath+'negative/'):
            os.makedirs(filepath+'negative/')  
        display_string=phase+' '+str(window_size)+' Storing Positive Indices'
        print(display_string)
        i=1
        n=len(pos_windows[index].keys())
        with open(filepath+'positive_ids.csv','w',newline='') as file:
            writer=csv.writer(file)
            for user_id in neg_windows[index]:
                print(display_string+': '+str(i)+'/'+str(n))
                windows=[gid+'-'+wid for gid,wid,_,_,_ in pos_windows[index][user_id]]
                writer.writerow([user_id]+windows)
                i+=1
        display_string=phase+' '+str(window_size)+' Storing Negative Indices'
        print(display_string)
        i=1
        n=len(neg_windows[index].keys())
        with open(filepath+'negative_ids.csv','w',newline='') as file:
            writer=csv.writer(file)
            for user_id in neg_windows[index]:
                print(display_string+': '+str(i)+'/'+str(n))
                windows=[gid+'-'+wid for gid,wid,_,_,_ in neg_windows[index][user_id]]
                writer.writerow([user_id]+windows)
                i+=1
        display_string=phase+' '+str(window_size)+' Storing Positive Windows Back Up'
        print(display_string)
        i=1
        n=len(pos_windows[index].keys())
        for user_id in pos_windows[index]:
            j=1
            m=len(pos_windows[index][user_id])
            with open('windows/'+index+'s/positive/'+user_id+'.csv','w',newline='') as file:
                writer=csv.writer(file)
                for gid,wid,start,end,gesture in neg_windows[index][user_id]:
                    print(display_string+': '+str(i)+'/'+str(n)+'   '+str(j)+'/'+str(m))
                    writer.writerow([gid,wid,start,end,gesture])
                    j+=1
            i+=1
        display_string=phase+' '+str(window_size)+' Storing Negative Windows Back Up'
        print(display_string)
        i=1
        n=len(neg_windows[index].keys())
        for user_id in neg_windows[index]:
            j=1
            m=len(neg_windows[index][user_id])
            with open('windows/'+index+'s/negative/'+user_id+'.csv','w',newline='') as file:
                writer=csv.writer(file)
                for gid,wid,start,end,gesture in neg_windows[index][user_id]:
                    print(display_string+': '+str(i)+'/'+str(n)+'   '+str(j)+'/'+str(m))
                    writer.writerow([gid,wid,start,end,gesture])
                    j+=1
            i+=1
        display_string=phase+' '+str(window_size)+' Storing Positive Features'
        print(display_string)
        i=1
        n=len(pos_windows[index].keys())
        for user_id in pos_windows[index]:
            j=1
            m=len(neg_windows[index][user_id])
            data1={}
            data2={}
            with open("data/filtered/"+user_id+'-gestures.csv')as f:
                temp=csv.reader(f)
                for datum in temp:
                    if datum[utils.g_index] in data1:
                        data1[datum[utils.g_index]].append(datum)
                    else:
                        data1[datum[utils.g_index]]=[datum]
            with open(filepath+'positive/'+user_id+'.csv','w',newline='') as file:
                writer=csv.writer(file)
                for gid,wid,start,end,gesture in pos_windows[index][user_id]:
                    print(display_string+': '+str(i)+'/'+str(n)+'   '+str(j)+'/'+str(m))
                    data=[datum for datum in data1[gid] if float(datum[utils.t_index])>=start and float(datum[utils.t_index])<=end]
                    features=extract_features.get_features(data)
                    if len(features)!=len(extract_features.feature_names):
                        raise NumFeaturesError(len(features))
                    if features is not None:
                        writer.writerow([gid+'-'+wid]+features)
                    j+=1
            i+=1
        display_string=phase+' '+str(window_size)+' Storing Negative Features'
        print(display_string)
        i=1
        n=len(neg_windows[index].keys())
        for user_id in neg_windows[index]:
            j=1
            m=len(neg_windows[index][user_id])
            data1={}
            data2={}
            with open("data/filtered/"+user_id+'-gestures.csv')as f:
                temp=csv.reader(f)
                for datum in temp:
                    if datum[utils.g_index] in data1:
                        data1[datum[utils.g_index]].append(datum)
                    else:
                        data1[datum[utils.g_index]]=[datum]
            try:
                with open("data/filtered/"+user_id+'-nongestures.csv')as f:
                    temp=csv.reader(f)
                    for datum in temp:
                        if datum[utils.g_index] in data2:
                            data2[datum[utils.g_index]].append(datum)
                        else:
                            data2[datum[utils.g_index]]=[datum]
            except Exception:
                pass
            with open(filepath+'negative/'+user_id+'.csv','w',newline='') as file:
                writer=csv.writer(file)
                for gid,wid,start,end,gesture in neg_windows[index][user_id]:
                    print(display_string+': '+str(i)+'/'+str(n)+'   '+str(j)+'/'+str(m))
                    if gesture:
                        data=[datum for datum in data1[gid] if float(datum[utils.t_index])>=start and float(datum[utils.t_index])<=end]
                    else:
                        data=[datum for datum in data2[gid] if float(datum[utils.t_index])>=start and float(datum[utils.t_index])<=end]    
                    features=extract_features.get_features(data)
                    if len(features)!=len(extract_features.feature_names):
                        raise NumFeaturesError(len(features))
                    if features is not None:
                        writer.writerow([gid+'-'+wid]+features)
                    j+=1
            i+=1
