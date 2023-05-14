#This module contains a GUI used to acquire the phase transition times for each gesture

import csv, math, os, re, shutil, sys
from turtle import color
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
        

def getFilePath(user_id,is_filtered=True):
    #generates the file path for the data from the user id and whether or not the data is filitered
    if is_filtered:
        return "data/filtered/"+user_id+"-gestures.csv"
    else:
        return "data/original/"+user_id+"-gestures.csv"

user_ids=["user001","user002","user003","user004","user005","user006","user007","user008","user009","user010","user011","user012","user013","user014","user015","user016"]
gesture_ids={}
sensors = ['Acc', 'Gyr', 'GRV', 'LAc']
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds, range from prewindowsize to 0)
x_index = 3 #x-value
y_index = 4 #y-value
z_index = 5 #z-value
w_index = 6 #GRV w-value
e_index = 7 #Euclidean norms of unfiltered values

#Generate a list of gesture ids
for user_id in user_ids:
    file_path=getFilePath(user_id)
    with open(file_path,'r') as file:
        data=list(csv.reader(file))
        data.pop(0)
        gesture_ids[user_id]=[]
        for datum in data:
            if datum[g_index] not in gesture_ids[user_id]:
                gesture_ids[user_id].append(datum[g_index])

def getData(user_id,gesture_id,is_filtered=True):
    #Get the data for a specified user and gesture
    file_path=getFilePath(user_id,is_filtered)
    with open(file_path,'r') as file:
        data=list(csv.reader(file))
        data.pop(0)
        to_return=[]
        for datum in data:
            if datum[g_index]==gesture_id:
                to_return.append(datum)
    return to_return

def graph(user_id,gesture_id,is_filtered=True):
    #Produce the graphs for the specified gesture
    data=getData(user_id,gesture_id,is_filtered)
    for i in range(4):
        sensor=sensors[i]
        times=[]
        x=[]
        y=[]
        z=[]
        w=[]
        e=[]
        for datum in data:
            if sensor == datum[s_index]:
                times.append(float(datum[t_index]))
                x.append(float(datum[x_index]))
                y.append(float(datum[y_index]))
                z.append(float(datum[z_index]))
                if 'GRV' == sensor:
                    w.append(float(datum[w_index]))
                else:
                    e.append(datum[e_index])
        n = len(times)
        plt.subplot(2,2,i+1)
        plt.title(sensor)
        plt.plot(times,x)
        plt.plot(times,y)
        plt.plot(times,z)
        if sensor=="GRV":
            plt.plot(times,w)
    plt.show()

class GUI(tk.Frame):

    def __init__(self,master=None):
        self.all_set=False
        #Create the GUI
        tk.Frame.__init__(self,master)
        #Create the empty graphs
        self.figure=plt.figure()
        self.plots={}
        for i in range(4):
            sensor=sensors[i]
            self.plots[sensor]=self.figure.add_subplot(2,2,i+1)
        self.canvas=FigureCanvasTkAgg(self.figure,master=root)
        self.canvas.get_tk_widget().grid(row=0,column=0,rowspan=10)
        #Initialise the user and gesture ids
        self.user_id_index=0
        self.user_id=user_ids[self.user_id_index]
        self.gesture_id_index=0
        self.gesture_id=gesture_ids[user_id][self.gesture_id_index]
        #Initialise a range of variables and GUI components
        self.changed=True
        self.user_label=tk.Label(master=root,text="User ID: "+self.user_id)
        self.user_label.grid(row=1,column=2,columnspan=2)
        self.gesture_label=tk.Label(master=root,text="Gesture ID: "+self.gesture_id)
        self.gesture_label.grid(row=2,column=2,columnspan=2)
        self.prev_user_button=tk.Button(master=root,text="Prev User",command=lambda: self.prevUser())
        self.prev_user_button.grid(row=4,column=2)
        self.next_user_button=tk.Button(master=root,text="Next User",command=lambda: self.nextUser())
        self.next_user_button.grid(row=4,column=3)
        self.prev_gesture_button=tk.Button(master=root,text="Prev Gesture",command=lambda: self.prevGesture())
        self.prev_gesture_button.grid(row=3,column=2)
        self.next_gesture_button=tk.Button(master=root,text="Next Gesture",command=lambda: self.nextGesture())
        self.next_gesture_button.grid(row=3,column=3)
        self.time_index=0
        self.times=self.getTimes() #Load the current transition times for the inital gesture
        #Initialise the time labels
        self.time_labels=[tk.Label(master=root,text="Time "+str(i+1)+": {:.2f}".format(self.times[i])) for i in range(4)]
        for i in range(4):
            self.time_labels[i].grid(row=5+i,column=2,columnspan=2)
        self.setTimeIndex(0)
        self.temp_time=0.0
        self.graph() #Fill in the graphs
        #Connect the mouse events to the relevant methods
        self.figure.canvas.mpl_connect("motion_notify_event",self.onMouseMove) 
        self.figure.canvas.mpl_connect("button_press_event",self.onMouseClick)

    def onMouseMove(self,event): #Update the current time label and the red line on the graphs when the mouse is moved over the graphs
        if event.inaxes is not None:
            self.temp_time=event.xdata
            self.time_labels[self.time_index].config(text="Time "+str(self.time_index+1)+": {:.2f}".format(self.temp_time))
            self.graph(vlines=event.xdata)

    def onMouseClick(self,event): #When the graphs are clicked fix the current time and select the next time to update
        if event.inaxes is not None:
            if self.time_index==3:
                self.all_set=True
            self.times[self.time_index]=self.temp_time
            self.setTimeIndex(self.time_index+1)

    def getTimes(self): #Loads the transition times for the current gesture
        with open(file_name,'r') as file:
            data=list(csv.reader(file))
        for datum in data:
            if (datum[user_index]==self.user_id) and (datum[gesture_index]==self.gesture_id):
                return [float(datum[i]) for i in range(2,6)]
        return [0.0]*4
        
    def graph(self,vlines=None,is_filtered=True): #Fill in the graphs
        if self.changed:
            self.data=getData(self.user_id,self.gesture_id,is_filtered)
            self.changed=False
            self.all_set=False
            self.setTimeIndex(0)
            self.times=self.getTimes()
            for i in range(4):
                self.time_labels[i].config(text="Time "+str(i+1)+": {:.2f}".format(self.times[i]))
        data=self.data
        for i in range(4):
            sensor=sensors[i]
            times=[]
            x=[]
            y=[]
            z=[]
            w=[]
            e=[]
            for datum in data:
                if sensor == datum[s_index]:
                    times.append(float(datum[t_index]))
                    x.append(float(datum[x_index]))
                    y.append(float(datum[y_index]))
                    z.append(float(datum[z_index]))
                    if 'GRV' == sensor:
                        w.append(float(datum[w_index]))
                    else:
                        e.append(datum[e_index])
            self.plots[sensor].clear()
            self.plots[sensor].title.set_text(sensor)
            self.plots[sensor].plot(times,x,linewidth=0.5)
            self.plots[sensor].plot(times,y,linewidth=0.5)
            self.plots[sensor].plot(times,z,linewidth=0.5)
            if sensor=="GRV":
                self.plots[sensor].plot(times,w,linewidth=0.5)
            if vlines is not None:
                self.plots[sensor].axvline(x=vlines,color="red",linewidth=0.5)
            for time in self.times:
                self.plots[sensor].axvline(x=time,color="black",linewidth=0.5)
        self.canvas.draw()

    def setTimeIndex(self,new_value): #Change the index of the time being updated
        self.time_index=new_value%4
        for i in range(4):
            self.time_labels[i].config(borderwidth=0,relief="flat")
        self.time_labels[self.time_index].config(borderwidth=2,relief="solid")

    def nextUser(self): #Select the first gesture of the next user
        if self.all_set:
            self.store()
        self.user_id_index=(self.user_id_index+1)%len(user_ids)
        self.user_id=user_ids[self.user_id_index]
        self.gesture_id_index=0
        self.gesture_id=gesture_ids[self.user_id][self.gesture_id_index]
        self.user_label.config(text="User ID: "+self.user_id)
        self.gesture_label.config(text="Gesture ID: "+self.gesture_id)
        self.changed=True
        self.graph()

    def prevUser(self): #Select the first gesture of the previous user
        if self.all_set:
            self.store()
        self.user_id_index=(self.user_id_index-1)%len(user_ids)
        self.user_id=user_ids[self.user_id_index]
        self.gesture_id_index=0
        self.gesture_id=gesture_ids[self.user_id][self.gesture_id_index]
        self.user_label.config(text="User ID: "+self.user_id)
        self.gesture_label.config(text="Gesture ID: "+self.gesture_id)
        self.changed=True
        self.graph()

    def nextGesture(self): #Select the next gesture of the current user
        if self.all_set:
            self.store()
        self.gesture_id_index=(self.gesture_id_index+1)%len(gesture_ids[self.user_id])
        self.gesture_id=gesture_ids[self.user_id][self.gesture_id_index]
        self.gesture_label.config(text="Gesture ID: "+self.gesture_id)
        self.changed=True
        self.graph()

    def prevGesture(self): #Select the previous gesture of the current user
        if self.all_set:
            self.store()
        self.gesture_id_index=(self.gesture_id_index-1)%len(gesture_ids[self.user_id])
        self.gesture_id=gesture_ids[self.user_id][self.gesture_id_index]
        self.gesture_label.config(text="Gesture ID: "+self.gesture_id)
        self.changed=True
        self.graph()

    def store(self): #Store the current transition times
        with open(file_name,'r') as file:
            data=list(csv.reader(file))
        for i in range(len(data)):
            if (data[i][user_index]==self.user_id) and (data[i][gesture_index]==self.gesture_id):
                data[i]=[self.user_id,self.gesture_id]+self.times
        with open(file_name,"w",newline='') as file:
            writer=csv.writer(file,dialect='excel')
            for datum in data:
                writer.writerow(datum)
        
#Initialise the times file
user_index=0
gesture_index=1
time_indices=[2,3,4,5]
file_name="times.csv"
if not os.path.exists(file_name):
    with open(file_name,"w",newline='') as file:
        writer=csv.writer(file,dialect='excel')
        for user_id in user_ids:
            for gesture_id in gesture_ids[user_id]:
                writer.writerow([user_id,gesture_id,0.0,0.0,0.0,0.0])

#Start the GUI
root=tk.Tk()
app=GUI(root)
app.mainloop()
