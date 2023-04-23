#This script generates histograms of the durations of each phase

import matplotlib.pyplot as plt
import csv

def get_duration(t1,t2):
    #Gets the duration based on the given start and end times
    #Returns None if the duration is less than 0.5 seconds
    d=float(t2)-float(t1)
    if d<0.5:
        d=None
    return d

def get_durations(times):
    #Gets the durations based on the list of given transition times
    d1=get_duration(times[0],times[1])
    d2=get_duration(times[1],times[2])
    d3=get_duration(times[2],times[3])
    return d1,d2,d3


#Read  and format the entries of the times file
time_file_path="times.csv"
with open(time_file_path,'r') as file:
    data=list(csv.reader(file))

times=[]

for datum in data:
    times.append(datum[2:])

#Initialise the list of durations
durations=[[],[],[]]

#Get all of the durations ignoring those less than 0.5 seconds
for t in times:
    d=get_durations(t)
    for i in range(3):
        if d[i] is not None:
            durations[i].append(d[i])

#Various arguments for the graph
plt.rcParams.update({'figure.figsize':(12,5)})
kwargs = dict(alpha=1, bins=100) 
labels=['Reaching','Alignment','Withdrawal']
fig,axs=plt.subplots(1,3)

#Create the graphs
for i in range(3):
    axs[i].hist(durations[i], **kwargs,)
    axs[i].set(ylabel='Frequency',xlabel='Duration',title=labels[i])
plt.show()