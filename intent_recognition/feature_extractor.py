#This script contains the feature extractor class which will extract features from the data
#The features considered are min, max, arithmetic mean, median, standard deviation, variance, interquartile range, kurtosis, skew, number of peaks, and various velocity and displacment stats.
#Each feature is extracted for each sensor and dimension (except for the velocity and displacement stats which are for each sensor except GRV)
#Note: the get_vel_disp_features function was taken from code produced by Jack Sturgess from the extract.py file of the repository: https://github.com/jacksturgess/watchauth

import utils
import math, statistics
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

#Create a dictionary of sensor dimension indices
sensor_dimension_indices={}

for sensor in utils.sensors:
    sensor_dimension_indices[sensor]=[utils.x_index,utils.y_index,utils.z_index,utils.e_index,None]

sensor_dimension_indices["GRV"]=[utils.x_index,utils.y_index,utils.z_index,utils.w_index]

def write_f_name(feature_fn):
    #Generates a list of feature names for a given feature function
    temp=[]
    s=feature_fn.name
    for sensor in feature_fn.sensors:
        dimensions=['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']
        if sensor=="GRV":
            dimensions=['x-', 'y-', 'z-', 'w-']
        for dimension in dimensions:
            temp.append(sensor+"-"+dimension+s)
    return temp

class FeatureFunction:
    #A class for feature functions with a method to produce an identical function that runs on a limited set of features

    def __init__(self,name,fn,sensors=utils.sensors):
        self.name=name
        self.fn=fn
        self.sensors=sensors

    def __call__(self, data):
        return self.fn(data)

    def limit(self,sensors=['Acc', 'Gyr']):
        return FeatureFunction(self.name,self.fn,sensors)

class FeatureFunction2(FeatureFunction):
    #Extends FeatureFunction with the ability to provide a function that generates the list of features when given a list of sensors

    def __init__(self,name,fn,get_names,sensors=utils.sensors):
        self.name=name
        self.fn=fn
        self.sensors=sensors
        self.names=get_names(sensors)
        self.get_names=get_names

    def limit(self,sensors=['Acc', 'Gyr']):
        return FeatureFunction2(self.name,self.fn,self.get_names,sensors)


class FeatureExtractor:
    #A class that will extract features
    #Key attribute: feature_names is the list of features that it extracts
    #Key Methods: get_features will extract features from the provided data

    def __init__(self,feature_fns=[],feature_fns2=[]):
        self.feature_names=[]
        for feature_fn in feature_fns:
            self.feature_names+=write_f_name(feature_fn)
        for feature_fn in feature_fns2:
            self.feature_names+=feature_fn.names
        self.feature_fns=feature_fns
        self.feature_fns2=feature_fns2
        self.current_user_id=None
        self.current_gesture_id=None
        self.current_gesture_data=[]
        self.current_is_filtered=None

    def load_gesture(self,user_id,gesture_id,gesture=True,is_filtered=True):
        #Load data for a gesture into the object
        self.current_user_id=user_id
        self.current_gesture_id=gesture_id
        self.current_is_filtered=is_filtered
        self.current_gesture_data=utils.getGestureData(user_id,gesture_id,gesture,is_filtered)
        if self.current_gesture_data is None:
            self.current_gesture_data=[]

    def get_interval(self,start_time,end_time):
        #Get all data from the currently loaded gesture that occurs in a given time interval
        temp=[]
        for datum in self.current_gesture_data:
            time=float(datum[utils.t_index])
            if (time>=start_time) and (time<=end_time):
                temp.append(datum)
        if len(temp)<45*len(utils.sensors):
            return None
        return temp

    def get_sensor_dimension_data(self,data,sensor,dim_index=None):
        #Filters the given data to return the data for a particular sensor and dimension.
        #If the dimension index is None then it returns the filtered euclidean norms for the specified sensor
        temp=[]
        for datum in data:
            if datum[utils.s_index]==sensor:
                if dim_index is not None:
                    temp.append(float(datum[dim_index]))
                else:
                    temp2=[]
                    for index in [utils.x_index,utils.y_index,utils.z_index]:
                        temp2.append(float(datum[index]))
                    temp.append(np.linalg.norm(temp2))
        return temp

    def get_sensor_data(self,data,sensor):
        #Filters the given data to return all data from the given sensor
        if data is None:
            return None
        temp=[]
        for datum in data:
            if datum[utils.s_index]==sensor:
                temp.append(datum)
        return temp

    def get_features(self,data):
        #Extracts features form the given data
        temp=[]
        for feature_fn in self.feature_fns:
            for sensor in feature_fn.sensors:
                for dim_index in sensor_dimension_indices[sensor]:
                    temp2=self.get_sensor_dimension_data(data,sensor,dim_index)
                    temp.append(feature_fn(temp2))
        for feature_fn in self.feature_fns2:
            for sensor in feature_fn.sensors:
                temp2=self.get_sensor_data(data,sensor)
                temp=temp+feature_fn(temp2)
        return temp

    def __call__(self,user_id,gesture_id,start_time,end_time,gesture=True,is_filtered=True):
        #Extracts the features for the specifed interval of the given gesture
        if (self.current_user_id!=user_id) or (self.current_gesture_id!=gesture_id) or (self.current_is_filtered!=is_filtered):
            self.load_gesture(user_id,gesture_id,gesture,is_filtered)
        data=self.get_interval(start_time,end_time)
        return self.get_features(data)

    def limit(self,sensors=['Acc', 'Gyr']):
        #Return a new object that runs on a limited number of sensors
        return FeatureExtractor([feature_fn.limit(sensors) for feature_fn in self.feature_fns],[feature_fn.limit(sensors) for feature_fn in self.feature_fns2])


def std(x): #Standard deviation
    return np.std(x,ddof=1)

def var(x): #Variance
    return np.var(x,ddof=1)

def iqr(x): #Interquartile range
    q75,q25=np.percentile(x, [75, 25])
    return q75-q25

threshold=0.5 #Threshold for peak counts

def pkcount(x): #Counts the number of peaks according to the above threshold
    temp=find_peaks(x,prominence=threshold)
    return len(temp)

#Create FeatureFunction objects for minimum, maximum, mean, median, standard deviation, variance, interquartile range, kurtosis, skew and peak count
feature_min=FeatureFunction("min",min)
feature_max=FeatureFunction("max",max)
feature_mean=FeatureFunction("mean",statistics.mean)
feature_median=FeatureFunction("median",statistics.median)
feature_stdev=FeatureFunction("standard-deviation",std)
feature_var=FeatureFunction("variance",var)
feature_iqr=FeatureFunction("interquartile-range",iqr)
feature_kurtosis=FeatureFunction("kurtosis",kurtosis)
feature_skew=FeatureFunction("skew",skew)
feature_pkcount=FeatureFunction("peak-count-"+str(threshold),pkcount)
features=[feature_min,feature_max,feature_mean,feature_median,feature_stdev,feature_var,feature_iqr,feature_kurtosis,feature_skew,feature_pkcount]

def get_names(sensors): #Return the list of feature names for the veocity and displacement statistics when given the sensor list
    temp=[]
    for sensor in sensors:
        for feature in ["-velocity-mean","-velocity-max","-displacement"]:
            for dimension in ["-x","-y","-z"]:
                temp.append(sensor+dimension+feature)
        temp.append(sensor+"-total-displacement")
    return temp

def get_vel_disp_features(data): 
    #Calculates the mean and maximum velocity and displacement along each of the x,y,z dimensions as well as the overall displacement
    #This function was taken from code produced by Jack Sturgess from the extract.py file of the repository: https://github.com/jacksturgess/watchauth

    temp_data=[[float(datum[utils.t_index]),float(datum[utils.x_index]),float(datum[utils.y_index]),float(datum[utils.z_index])] for datum in data]

    vx = [0]
    dx = [0]
    vy = [0]
    dy = [0]
    vz = [0]
    dz = [0]
    d = [0]
    n = len(temp_data) - 1 #number of samples
    dt = float((temp_data[n][0] - temp_data[0][0]) / n) #sample interval
    for j in range(n):
        vx.append(vx[j] + (temp_data[j][1] + temp_data[j + 1][1]) / 2 * dt / 10)
        dx.append(dx[j] + vx[j + 1] * dt / 10)
        vy.append(vy[j] + (temp_data[j][2] + temp_data[j + 1][2]) / 2 * dt / 10)
        dy.append(dy[j] + vy[j + 1] * dt / 10)
        vz.append(vz[j] + (temp_data[j][3] + temp_data[j + 1][3]) / 2 * dt / 10)
        dz.append(dz[j] + vz[j + 1] * dt / 10)
        d.append(math.sqrt(dx[j] * dx[j] + dy[j] * dy[j] + dz[j] * dz[j]))
    vx.pop(0)
    vy.pop(0)
    vz.pop(0)

    f=[]
    f.append(sum(vx) / len(vx))
    f.append(sum(vy) / len(vy))
    f.append(sum(vz) / len(vz))
    f.append(max(vx, key = abs))
    f.append(max(vy, key = abs))
    f.append(max(vz, key = abs))
    f.append(dx[len(dx) - 1])
    f.append(dy[len(dy) - 1])
    f.append(dz[len(dz) - 1])
    f.append(d[len(d) - 1])

    return f

#Create a FeatureFunction object for the velocity and displacement statistics that runs on all sensors except for GRV
feature_vel_disp=FeatureFunction2("velocity-displacement",get_vel_disp_features,get_names,[sensor for sensor in utils.sensors if sensor!="GRV"])

#Create a FeatureExtractor object
extract_features=FeatureExtractor(features,[feature_vel_disp])

#Create a FeatureExtractor object that only extracts features from the accelerometer and gyroscopic sensors
extract_limited_features=extract_features.limit()