
import torch
import numpy as np
import pandas as pd
import os


import Mining

import re
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')


from sklearn.utils import shuffle



class TNN(nn.Module):
    def __init__(self,intputdim):
        super(TNN, self).__init__()
        self.dense1 = nn.Linear(intputdim, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 128)
        self.relu = nn.ReLU() 

    def forward(self, x):
     
        x= self.relu(self.dense1(x))
        x=self.relu(self.dense2(x))
        x=self.relu(self.dense3(x))
        x=self.relu(self.dense4(x))

        out=self.dense5(x)
        out=F.normalize(out)
        return out



def reorganize(x, y):
    
    """ 
    reorganize data  
    transform to dict 
    """
    assert x.shape[0] == y.shape[0]
    """ 0, 2, 4, 8, 16, 32 represent Embryonic cells and 2 cells 4cells 8cells····  respectively in demo data"""
    dataset = {i: [] for i in [0,2,4,8,16,32]}
    for i in range(x.shape[0]):
        dataset[y[i]].append(x[i])
        
    return dataset   

def makelabel(label):
    """ 
    make label
    transform '2cell' -> 2  '4cell' -> 4················
    """
    cell_label=[]
    for i in label:
        if((re.findall('zygote',i)==['zygote']) | (re.findall('zy',i)==['zy']) | (re.findall('Zy',i)==['Zy']) | (re.findall('early2cell',i)==['early2cell'])):
            cell_label.append(0)
           # cell_label.append('Zygote')
            continue
        
        if( (re.findall('\d+cell',i)==['2cell']) | (re.findall('2-cell',i)==['2-cell'])):
            cell_label.append(2)
            continue

        if( (re.findall('\d+cell',i)==['4cell']) | (re.findall('4-cell',i)==['4-cell'])  ):
            cell_label.append(4)
            continue
        
        if( (re.findall('\d+cell',i)==['8cell']) | (re.findall('8-cell',i)==['8-cell']) ):
            cell_label.append(8)
            continue
        
        if( (re.findall('\d+cell',i)==['16cell']) ):
            cell_label.append(16)
            continue
        
        if( (re.findall('\d+cell',i)==['32cell']) | (re.findall('blast',i)==['blast'])):
            cell_label.append(32)
            continue
        
    return cell_label




def makedata(txt,shuffle_flag):
    
    
    data = pd.read_csv(txt,sep = ',',header = 0)
    data = data.set_index('gene')

    if(shuffle_flag):
        data = (shuffle(data.T)).T 
        
    X = data.values.T  

    """
    Do a log transformation on x
    """
    tmp=np.unique(X)
    tmp=tmp[tmp!=0]
    tmp_min=np.min(tmp)
    add_item=tmp_min*1e-3
    X = np.log(X+add_item)   

    """ rescale 0-1 """
    for i in range(X.shape[0]):
        X[i,:] =  X[i,:] - np.min(X[i,:])  
        X[i,:] =  X[i,:] / np.max(X[i,:])
   
    y_sample = data.columns.values.tolist()  
     

    """ 
    make label
    transform '2cell' -> 2  '4cell' -> 4················
    """
    cell_label=makelabel(y_sample)

    return X,cell_label



def train_loop_KFold(  x_train_set,label_train_set):
    # set the model in training mode
    model.train()
    
    running_loss = 0.0
    
    # get the inputs
    x_train,y_train = x_train_set,label_train_set 
    x_train=torch.tensor(x_train,device=device)
    y_train=torch.tensor(y_train,device=device)

    # zero the parameter gradients
    optimizer.zero_grad()
    
    """ forward + backward + optimize """
    outputs = model(x_train.float()) 

    # construct train triplets and online mining, compute triplet loss
    loss, pos_triplet,pos_triplet_no_margin, valid_triplet , max_pair_dist = Mining.online_mine_all(labels=y_train, embeddings=outputs, margin=margin,dist_model= dist_model, squared=True, device=device)
    
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    print(f"train_loss = {loss:.5f}, trainable_triplet = {pos_triplet},trainable_triplet_no_margin = {pos_triplet_no_margin}, total_triplets = {valid_triplet} ")
   
    return running_loss
    

def test_loop_KFold(x_val_set,label_val_set):
    # set the model in predict mode
    model.eval()
    with torch.no_grad():
        test_loss=0.0
        
        test_data=torch.tensor(x_val_set,device=device)
        test_label=torch.tensor(label_val_set,device=device)

        predict = model(test_data.float())
        # construct test triplets and online mining, compute triplet loss
        test_loss, pos_triplet, pos_triplet_no_margin,valid_triplet,max_pair_dist = Mining.online_mine_all(labels=test_label, embeddings=predict, margin=margin,dist_model=dist_model, squared=True, device=device)

        return test_loss



def train_KFoldVal(K):
    
    """ K-fold cross-validation """
    from sklearn.model_selection import StratifiedKFold,KFold
    skf = StratifiedKFold(n_splits= K)
    min_test_loss=10000
    for epoch in range(epoch_num):  # loop over the dataset multiple times
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_loss_sum = 0
            val_loss_sum = 0
            for fold,(train_idx,val_idx) in enumerate(skf.split(X_train,cell_train_label)):

                x_train_set,x_val_set = X_train[train_idx],X_train[val_idx]

                label_train_set,label_val_set = np.array(cell_train_label)[train_idx],np.array(cell_train_label)[val_idx]

                epoch_loss=train_loop_KFold( x_train_set,label_train_set)
                epoch_loss_sum += epoch_loss
            
                epoch_val_loss=test_loop_KFold(x_val_set,label_val_set)
                val_loss_sum += epoch_val_loss

            train_loss=epoch_loss_sum/K
            val_loss = val_loss_sum/K
            print(f"At epoches = {epoch+1}\n epoch_loss = {train_loss}")
            print(f"Validation  loss: \n {val_loss:>8f} \n")

            scheduler.step(val_loss)
            
            if( not os.path.exists("model")):
                os.makedirs("model")

            if(save_best_model): 
                """ save the best model base on val_loss """
                if(val_loss < min_test_loss):
                    min_test_loss = val_loss
                    early_stop_num=0
                    torch.save(model.state_dict(),best_model_path)
                    print('val_loss improved , save model')
                else:
                    early_stop_num +=1
                    if(early_stop_num>= early_stop_patience):
                        print('model didnt improve during '+str(early_stop_patience)+' epoch, earlystop with the best val_loss:'+str(min_test_loss))
                        break
            else:
                """ save every epoch model during training """
                torch.save(model.state_dict(),f'{model_path}epoch:{epoch}.pth')




if __name__ == "__main__":

    

    """ set Hyper parameter to train """
    """ 
    Parameters
    ----------
    device : 'str' , choose cpu or gpu train the model
    margin : 'float', a number in triplet loss function : max( d(a,p) - d(a,n) + margin , 0 )
    save_best_model : 'boolean', 'true' save the best model , 'false' save every epoch model during training
    best_model_path : 'str' , the model save path
    model_path : 'str' , save path of every epoch model

    K : 'int', K-fold cross-validation

    early_stop_patience : 'int', stop training early when loss do not decline after early_stop_patience epoch
    epoch_num : 'int', training epoch

    lr_init : 'float', Initial learning rate
    lr_factor : 'float', new_learning_rate = old_learning_rate * lr_factor
    lr_patience : 'int', Reduce the learning rate when loss do not decline after lr_patience epoch
    
    dist_model : 'str' choose which distance to ues , 'cos' represent cosine distance ,  'euclidean distances'

    """
    # device = torch.device('cuda:0') 
    device = 'cpu'
    margin=0.03   
    save_best_model= False
    K=4 
    early_stop_patience=50
    epoch_num=150
    lr_init = 0.0001
    lr_factor = 0.5
    lr_patience = 10
    dist_model= 'cos'       
    best_model_path='model/best_model.pth'
    model_path = 'model/'

    train_dataset_path = 'data/train.csv'
    test_dataset_path = 'data/test.csv'

    """ get data  """
    X_train,cell_train_label= makedata(train_dataset_path,shuffle_flag=True)
    X_test,cell_test_label= makedata(test_dataset_path,shuffle_flag=True)
    
    """ reorganize dataset """
    train_set=reorganize(X_train,np.asarray(cell_train_label))
    test_set=reorganize(X_test,np.asarray(cell_test_label))

    """ create triplet neural network """
    model=TNN(intputdim=X_train.shape[1]).to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr= lr_init )                       

    
    """ pytorch learning rate scheduler
    Reduce the learning rate when val loss do not decline """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor= lr_factor, patience=lr_patience, verbose=True)
    
    """ start training """
    train_KFoldVal(K)
        
    print('Finished Training')

