import torch
import numpy as np
import pandas as pd
import Mining
import os
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')



class TNN(nn.Module):
    def __init__(self,intputdim):
        super(TNN, self).__init__()
        self.dense1 = nn.Linear(intputdim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense_supplement1 = nn.Linear(512, 512)
        self.dense_supplement2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 128)

        self.drop08 = nn.Dropout(p= 0.8) 
        self.drop07 = nn.Dropout(p= 0.7) 
        self.drop06 = nn.Dropout(p= 0.6) 
        self.drop05 = nn.Dropout(p= 0.5) 
        self.drop04 = nn.Dropout(p= 0.4) 
        self.drop03 = nn.Dropout(p= 0.3)
        self.drop02 = nn.Dropout(p= 0.2) 
        self.drop01 = nn.Dropout(p= 0.1) 
        
        self.relu = nn.ReLU() 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x= self.relu(self.dense1(x))
        

        x=self.relu(self.dense2(x))
        x=self.relu(self.dense3(x))
        x=self.relu(self.dense4(x))
        out=self.dense5(x)
        out=F.normalize(out)
        return out

def train_loop(train_data,train_label,epoch):
    # set the model in training mode
    model.train()
    epoch_loss = 0.0
    # get the inputs
    x_train,y_train = train_data,train_label 

    x_train=torch.tensor(x_train,device=device)
    y_train=torch.tensor(y_train,device=device)
    """ zero the parameter gradients """
    optimizer.zero_grad()
    """ forward + backward + optimize """
    outputs = model(x_train.float()) 
    # construct train triplets and online mining, compute triplet loss
    loss, pos_triplet,pos_triplet_no_margin, valid_triplet , max_pair_dist = Mining.online_mine_all(labels=y_train, embeddings=outputs, margin=margin,dist_model= dist_model, squared=True, device=device)

    # Update the network weights
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()

    print(f"trainable_triplet = {pos_triplet},trainable_triplet_no_margin = {pos_triplet_no_margin}, total_triplets = {valid_triplet} ")
    print(f"At epoches = {epoch+1}\n epoch_loss = {epoch_loss}")
    return epoch_loss


def test_loop(test_data,test_label,remove_num):

    # set the model in predict mode
    model.eval()
    with torch.no_grad():
        test_loss=0.0
        test_data=torch.tensor(test_data,device=device) 
        test_label=torch.tensor(test_label,device=device)

        predict = model(test_data.float())

        # construct test triplets and online mining, compute triplet loss
        # Note that in this experiment, test triplets refer to those that contain testing samples specified by the 'remove_num'
        test_loss, pos_triplet, pos_triplet_no_margin,valid_triplet,max_pair_dist = Mining.online_mine_all(labels=test_label, embeddings=predict, margin=margin,dist_model=dist_model,test=remove_num, squared=True, device=device)
        print(f"Test loss:{test_loss:>8f}, test_trainable_triplet = {pos_triplet}, test_trainable_triplet_no_margin = {pos_triplet_no_margin}, test_total_triplets = {valid_triplet}\n")
        
        scheduler.step(test_loss)
        return test_loss


def train_test(train_data,train_label,all_data,all_label,remove_num):
    
    min_test_loss=10000

    for epoch in range(epoch_num):  
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_loss=train_loop(train_data,train_label, epoch)
            epoch_loss_list.append(epoch_loss)
            test_loss=test_loop(all_data,all_label,remove_num)

            if( not os.path.exists("model")):
                os.makedirs("model")

            if(save_best_model):
                if(test_loss < min_test_loss):
                    min_test_loss = test_loss
                    early_stop_num=0
                    torch.save(model.state_dict(),best_model_path)
                    print('test_loss improved , save model')
                else:
                    early_stop_num +=1
                    if(early_stop_num>= early_stop_patience):
                        print('model didnt improve during '+str(early_stop_patience)+' epoch, earlystop with the best test_loss:'+str(min_test_loss))
                        break
            else:
           
                torch.save(model.state_dict(),f'{model_path}epoch:{epoch}.pth')




def get_data_otu(file_name,remove):
    """ if 'remove' is empty , return complete data 
        if 'remove' is not empty ,  remove sample which specified by the 'remove' and return remaining samples
        the 'label_num' is just a list:[1,2,3,4·········] that represent longitudinal trend of data
        """
    data = pd.read_csv(file_name,sep= ',', index_col=0)

    if (len(remove)>0):
        remove = [i-1 for i in remove]
        data = data.drop(data.columns[remove],axis=1)
        
    X = data.values.T   
    for i in range(X.shape[0]): 
        # do log transform
        X[i,:] = np.log( (X[i,:] /  X[i,:].sum()*10000)+1 )

    label_num= list(range(1,X.shape[0]+1))
    return X,label_num






""" set Hyper parameter to train """
""" 
Parameters
----------
device : 'str' , choose cpu or gpu train the model
margin : 'float', a number in triplet loss function : max( d(a,p) - d(a,n) + margin , 0 )
save_best_model : 'boolean', 'true' save the best model , 'false' save every epoch model during training
best_model_path : 'str' , save path of the best model
model_path : 'str' , save path of every epoch model

K : 'int', K-fold cross-validation

early_stop_patience : 'int', stop training early when loss do not decline after early_stop_patience epoch
epoch_num : 'int', training epoch

lr_init : 'float', Initial learning rate
lr_factor : 'float', new_learning_rate = old_learning_rate * lr_factor
lr_patience : 'int', Reduce the learning rate when loss do not decline after lr_patience epoch

dist_model : 'str', choose which distance to ues , 'cos' represent cosine distance ,  'euclidean distances'

remove_list : 'list', choose one or two samples as unknown testing data from temporal points 
                [4,10] represent choose the fourth and tenth samples as unknown testing data from 30 temporal points in our article, for example
dataset_path : 'str' the path to get data for training
"""
# device = torch.device('cuda:0')
device = 'cpu' 
margin=0.5
save_best_model= False
best_model_path='model/best_model.pth'
model_path = 'model/'

early_stop_patience=50
epoch_num=100

lr_init = 0.0001
lr_factor = 0.5
lr_patience = 20

dist_model= 'cos'        #'cos'  or 'euclidean distances'  
remove_list = [4]  
dataset_path = "data/EP584156_rectum.csv"

if __name__ == "__main__":
    epoch_loss_list = []

    # get data
    all_data,all_label=get_data_otu(file_name=dataset_path,remove=[]) 
    train_data,train_label=get_data_otu(file_name=dataset_path,remove=remove_list)
    

    """ create triplet neural network """
    model=TNN(intputdim=train_data.shape[1]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr= lr_init)     

    """ pytorch learning rate scheduler
    Reduce the learning rate when val loss do not decline """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor= lr_factor, patience=lr_patience, verbose=True)

    train_test(train_data,train_label,all_data,all_label,remove_list)
    
    print('Finished Training')

