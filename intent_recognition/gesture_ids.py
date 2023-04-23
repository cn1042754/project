#This script creates and stores lists of the gesture ids for each user to speed up initialisation of the utils script


import csv,os

#List of user ids
user_ids=["user001","user002","user003","user004","user005","user006","user007","user008","user009","user010","user011","user012","user013","user014","user015","user016"]

#index values for the data files
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds, range from prewindowsize to 0)
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

#file names to store the ids in
file_path1="gesture_ids.csv"
file_path2="nongesture_ids.csv"

#Initialise dictionaries to store the gesture ids
gesture_ids={}
nongesture_ids={}

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

#Get the gesture ids from the relevant files (the gesture ids can be gathered from the times file as the phase_splitter script automatically gathers all gesture ids)    

for user_id in user_ids:
    print(user_id)
    gesture_ids[user_id]=[]
    if os.path.exists(getDataFilePath(user_id,False)):
        nongesture_ids[user_id]=[]
        file_path=getDataFilePath(user_id,False)
        with open(file_path,'r') as file:
            data=list(csv.reader(file))
            data.pop(0)
            for datum in data:
                gesture_id=datum[g_index]
                if (len(nongesture_ids[user_id])==0) or (gesture_id!=nongesture_ids[user_id][-1]):
                    nongesture_ids[user_id].append(gesture_id)

time_file_path="times.csv"
with open(time_file_path,"r") as file:
    data=list(csv.reader(file))
for datum in data:
    gesture_ids[datum[tu_index]].append(datum[tg_index])

#Store the gesture ids

with open(file_path1,'w',newline='') as file:
    writer=csv.writer(file)
    for user_id in gesture_ids:
        writer.writerow([user_id]+gesture_ids[user_id])

with open(file_path2,'w',newline='') as file:
    writer=csv.writer(file)
    for user_id in nongesture_ids:
        writer.writerow([user_id]+nongesture_ids[user_id])

