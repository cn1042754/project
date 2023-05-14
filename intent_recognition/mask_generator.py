#This script contains a class that generates the training, validation and test masks

import utils,os,math,csv
import numpy as np

default_num_splits=5

class MaskGenerator:

    def __init__(self,name,n_splits=default_num_splits,test=None):
        np.random.seed(123) #Manually seed the random generator to get repeatable results
        #Initialise the varaibles
        self.n_splits=n_splits
        self.name=name
        self.test_indexing=True
        if test is None:
            self.user=False
            self.uid=""
            self.terminal=False
            self.tid=""
        elif test[:4]=="user":
            self.user=True
            self.uid=test
            if len(utils.nongesture_ids[self.uid])>0:
                self.test_indexing=False
            self.terminal=False
            self.tid=""
            self.n_splits-=1
        else:
            self.terminal=True
            self.tid=test
            self.user=False
            self.uid=""
            self.n_splits-=1
        path='masks/'+name+'/'
        if not os.path.exists(path):
            print(name)
            os.mkdir(path)
            #Split the tap gesture data
            for uid in [x for x in utils.user_ids if x!=self.uid]:
                gids=utils.gesture_ids[uid]
                if not self.terminal:
                    n=len(gids)
                    n_per_split=math.floor(n/self.n_splits)
                    diff=n-n_per_split*self.n_splits
                    mask=np.concatenate([np.array([i]*n_per_split) for i in range(self.n_splits)]+[np.array([0]*diff)])
                    np.random.shuffle(mask)
                    splits=[[gids[i] for i in range(len(gids)) if mask[i]==j] for j in range(self.n_splits)]
                else:
                    testing=[gid for gid in gids if gid[:4]=="TAP"+self.tid]
                    to_split=[gid for gid in gids if gid[:4]!="TAP"+self.tid]
                    n=len(to_split)
                    n_per_split=math.floor(n/self.n_splits)
                    diff=n-n_per_split*self.n_splits
                    mask=np.concatenate([np.array([i]*n_per_split) for i in range(self.n_splits)]+[np.array([0]*diff)])
                    np.random.shuffle(mask)
                    splits=[[gids[i] for i in range(n) if mask[i]==j] for j in range(self.n_splits)]
                    file_path=path+'test-'+uid+'-gestures.csv'
                    self.store(testing,file_path)
                for i in range(len(splits)):
                    file_path=path+'split'+str(i)+'-'+uid+'-gestures.csv'
                    self.store(splits[i],file_path)
            if self.user:
                gids=utils.gesture_ids[self.uid]
                testing=gids
                file_path=path+'test-'+uid+'-gestures.csv'
                self.store(testing,file_path)
            #Split the non-gesture data
            if not self.user or (self.user and len(utils.nongesture_ids[self.uid])==0):
                if (self.user and len(utils.nongesture_ids[self.uid])==0) or self.terminal:
                    num_splits=n_splits
                else:
                    num_splits=self.n_splits
                for uid in [uid for uid in utils.user_ids if len(utils.nongesture_ids[uid])>0]:
                    gids=utils.nongesture_ids[uid]
                    n=len(gids)
                    n_per_split=math.floor(n/num_splits)
                    diff=n-n_per_split*num_splits
                    mask=np.concatenate([np.array([i]*n_per_split) for i in range(num_splits)]+[np.array([0]*diff)])
                    np.random.shuffle(mask)
                    splits=[[gids[i] for i in range(len(gids)) if mask[i]==j] for j in range(num_splits)]
                    if (self.user and len(utils.nongesture_ids[self.uid])==0) or self.terminal:
                        testing=splits[0]
                        splits=splits[1:]
                        file_path=path+'test-'+uid+'-nongestures.csv'
                        self.store(testing,file_path)
                    for i in range(len(splits)):
                        file_path=path+'split'+str(i)+'-'+uid+'-nongestures.csv'
                        self.store(splits[i],file_path)
            else:
                for uid in [uid for uid in utils.user_ids if uid!=self.uid]:
                    gids=utils.nongesture_ids[uid]
                    n=len(gids)
                    n_per_split=math.floor(n/self.n_splits)
                    diff=n-n_per_split*self.n_splits
                    mask=np.concatenate([np.array([i]*n_per_split) for i in range(self.n_splits)]+[np.array([0]*diff)])
                    np.random.shuffle(mask)
                    splits=[[gids[i] for i in range(len(gids)) if mask[i]==j] for j in range(self.n_splits)]
                    for i in range(len(splits)):
                        file_path=path+'split'+str(i)+'-'+uid+'-nongestures.csv'
                        self.store(splits[i],file_path)
                gids=utils.nongesture_ids[self.uid]
                testing=gids
                file_path=path+'test-'+self.uid+'-nongestures.csv'
                self.store(testing,file_path)

    def store(self,gids,path):
        #Store the given gesture ids in the specified file
        with open(path,'w',newline='') as file:
            writer=csv.writer(file)
            for gid in gids:
                writer.writerow([gid])

    def read(self,path):
        #Read the gesture ids from a csv file
        try:
            with open(path,'r') as file:
                data=csv.reader(file)
                temp=[datum[0] for datum in data]
            return temp
        except Exception:
            return []

    def get_mask(self,name):
        #Retrieves the gesture and non-gesture ids for the specified mask or split
        gestureids={}
        path='masks/'+self.name+'/'+name
        for uid in utils.user_ids:
            file_path=path+'-'+uid
            gestureids[uid]=self.read(file_path+'-gestures.csv')
            temp=self.read(file_path+'-nongestures.csv')
            gestureids[uid]+=temp
        return gestureids

    def combine(self,splits):
        #Combines a list of splits
        to_return={}
        for split in splits:
            for uid in split:
                if uid in to_return:
                    to_return[uid]+=split[uid]
                else:
                    to_return[uid]=split[uid]
        return to_return

    def get_splits(self):
        #Gets a list of the splits
        splits=[]
        for i in range(self.n_splits):
            splits.append(self.get_mask('split'+str(i)))
        return splits

    def get_train(self,val_split_index=0,test_split_index=-1):
        #Retrieves the training mask for the given validation and test split indices
        splits=self.get_splits()
        if self.test_indexing and not self.user and not self.terminal:
            splits=[splits[i] for i in range(len(splits)) if (i!=test_split_index)]
        splits_train=[splits[i] for i in range(len(splits)) if (i!=val_split_index)]
        return self.combine(splits_train)

    def get_validation(self,val_split_index=0,test_split_index=-1):
        #Retrieves the validation mask for the given validation and test split indices
        if not self.test_indexing or self.user or self.terminal:
            return self.get_mask('split'+str(val_split_index))
        splits=self.get_splits()
        splits_no_test=[splits[i] for i in range(len(splits)) if (i!=test_split_index)]
        return splits_no_test[val_split_index]

    def get_testing(self,test_split_index=0):
        #Retrieves the test mask for the given validation and test split indices
        if self.user:
            return self.get_mask('test')
        elif self.terminal:
            return self.combine([self.get_mask('test'),self.get_mask('split'+str(test_split_index))])
        else:
            return self.get_mask('split'+str(test_split_index))
        
#Create the general mask
general_mask=MaskGenerator("General")
#For each user create a mask  using that user's data as the test data
user_masks={}
for uid in utils.user_ids:
    user_masks[uid]=MaskGenerator("User mask-"+uid,test=uid)
#For each terminal create a mask using that terminal's data as the test data
terminal_masks={}
for tid in utils.terminal_ids:
    terminal_masks[tid]=MaskGenerator("Terminal mask-"+tid,test=tid)
