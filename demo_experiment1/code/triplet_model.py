# -*- coding: utf-8 -*-

import sys
import optparse
import os
import random
import numpy as np
import pandas as pd

#if you dont want to use CPU,you can delete it.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import defaultdict
# import keras
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input, Dense, Lambda
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight
os.environ['KERAS_BACKEND']='tensorflow'
np.random.seed(123) 

def list_duplicates(seq,index):    
    tally = defaultdict(list) 
    for i,item in enumerate(seq): 
        if i in index and item != '':
            tally[item].append(i)   
    return ((key,locs) for key,locs in tally.items() if len(locs)>1) 

#creat triplet
def triplet(level,all_level,file_name,index):
    anchor = []
    positive = []
    negative = []
    non_unique_train = []
    new_index = []
 
    for dup in sorted(list_duplicates(all_level[0],index)):
        if level != all_level[0]:
            new_index.extend( random.sample(dup[1], 1))
        else:
            neiw_index = index
    label = []  
    for dup in sorted(list_duplicates(level,new_index)):    
        non_unique_train.append(dup)
    for iterm in non_unique_train:
        for anchor_iterm in iterm[1]:
            positive_index = iterm[1][:]
            positive_index.remove(anchor_iterm)
            for positive_iterm in positive_index:
                negative_index =   list( (set(new_index) | set(iterm[1])) - (set(new_index) & set(iterm[1])))                
                for negative_iterm in negative_index:
                    #Triplet sampling is used in the class layer. If you have too many samples, you can also use this method in the genes, family and order layers.
                    if level == all_level[3]:
                        if all_level[2][anchor_iterm] != all_level[2][positive_iterm] and all_level[4][positive_iterm] == all_level[4][negative_iterm] != '':
                            anchor.append(file_name[anchor_iterm])
                            positive.append(file_name[positive_iterm])
                            negative.append(file_name[negative_iterm])
                            for i in range(6):
                                if all_level[i][anchor_iterm] == all_level[i][positive_iterm]:
                                    same_po = i+1
                                    break
                            for i in range(6):
                                 if all_level[i][anchor_iterm] == all_level[i][negative_iterm]:
                                      different_po = i+1
                                      break
                            label.append(different_po-same_po)
                    else:
                            anchor.append(file_name[anchor_iterm])
                            positive.append(file_name[positive_iterm])
                            negative.append(file_name[negative_iterm])
             
                            for i in range(6):
                                if all_level[i][anchor_iterm] == all_level[i][positive_iterm]:
                                    same_po = i+1
                                    break
                            for i in range(6):
                                 if all_level[i][anchor_iterm] == all_level[i][negative_iterm]:
                                      different_po = i+1
                                      break
                 
                            label.append(different_po-same_po)
      
    return anchor,positive,negative,label

#Convert species names into corresponding k-mer vectors
def list2data(list_file, kmer_data, k):
    file_number = len(list_file)
    data = np.zeros((file_number, 4**k),dtype='float32')
    for i in range(len(list_file)):
        data[i] = (kmer_data[list_file[i]])
    return data


def shuffle(a,b,c,d): 
     multi_list = list(zip(a, b, c,d))
     random.shuffle(multi_list)     
     a[:], b[:],c[:] ,d[:]=  zip(*multi_list) 
     return a,b,c,d

# compute loss, x is anchor，y is positive, z is negtive
def eulidean_distance(vects):
    x,y,z = vects
    sum_square = K.sqrt(K.sum(K.square(x-y),axis=1,keepdims=True))-K.sqrt(K.sum(K.square(x-z),axis=1,keepdims=True))+0.5
    return K.maximum(sum_square,0)

#This is the model output value because there is no y_label
def identity_loss(y_true,y_pred):   
    return K.mean(y_pred-0*y_true)

#Construction of triplet model
def triplet_model(k):
    query_input = Input(shape=(4**k, ), name='Query_Input', dtype='float32')
    positive_input = Input(shape=(4**k,), name='Positive_Input', dtype='float32')
    negative_input = Input(shape=(4**k,), name='Negative_Input', dtype='float32')


    dense1 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_1")                                                                                               
    dense2 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_2")                                                                                               
    dense3 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_3")                                                                                               
    dense4 = Dense(200, activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_4")                                                                                              
    dense5 = Dense(200,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_5")                                                                                                
    dense6 = Dense(200,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_6")                                                                                              
    dense7 = Dense(200,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_7")                                                                                              
    dense8 = Dense(100,activation='relu', use_bias=True, kernel_initializer="glorot_normal",name="Dense_8")
    dense20 = Dense(100, use_bias=True, kernel_initializer="glorot_normal",name="Dense_20")

    que_dense1 = dense1(query_input)
    pos_dense1 = dense1(positive_input)
    neg_dense1 = dense1(negative_input)
   
    que_dense2 = dense2(que_dense1)
    pos_dense2 = dense2(pos_dense1)
    neg_dense2 = dense2(neg_dense1)
    
    que_dense3 = dense3(que_dense2)
    pos_dense3 = dense3(pos_dense2)
    neg_dense3 = dense3(neg_dense2)
        
    que_dense4 = dense4(que_dense3)
    pos_dense4 = dense4(pos_dense3)
    neg_dense4 = dense4(neg_dense3)
   
         
    que_dense5 = dense5(que_dense4)
    pos_dense5 = dense5(pos_dense4)
    neg_dense5 = dense5(neg_dense4)
    
    que_dense6 = dense6(que_dense5)
    pos_dense6 = dense6(pos_dense5)
    neg_dense6 = dense6(neg_dense5)
    
    que_dense7 = dense7(que_dense6)
    pos_dense7 = dense7(pos_dense6)
    neg_dense7 = dense7(neg_dense6)
    
    que_dense8 = dense8(que_dense7)
    pos_dense8 = dense8(pos_dense7)
    neg_dense8 = dense8(neg_dense7)

    que_out = dense20(que_dense8)
    pos_out = dense20(pos_dense8)
    neg_out = dense20(neg_dense8)
    
    triplet_loss = Lambda(eulidean_distance, 
                  output_shape = (1,))([que_out,pos_out,neg_out])
    model = Model(inputs = [query_input, positive_input, negative_input],outputs = triplet_loss)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=identity_loss, optimizer=adam)
    return model

#Construct the triplets at layers genus, family, order and class, respectively
def train_test_triplet(kingdom,phylum,class_,order,family,genus,file_name,index):
    anchor_list_class,positive_list_class,negative_list_class,label_class = triplet(class_,[genus,family,order,class_,phylum,kingdom],file_name,index)
    anchor_list_order,positive_list_order,negative_list_order,label_order = triplet(order,[genus,family,order,class_,phylum,kingdom],file_name,index)
    anchor_list_family,positive_list_family,negative_list_family,label_family = triplet(family,[genus,family,order,class_,phylum,kingdom],file_name,index)
    anchor_list_genus,positive_list_genus,negative_list_genus,label_genus = triplet(genus,[genus,family,order,class_,phylum,kingdom],file_name,index)
   
    anchor_list = anchor_list_class + anchor_list_order + anchor_list_family + anchor_list_genus
    positive_list = positive_list_class + positive_list_order + positive_list_family + positive_list_genus
    negative_list = negative_list_class + negative_list_order + negative_list_family + negative_list_genus
    distance_label = label_class + label_order + label_family + label_genus

    return anchor_list,positive_list,negative_list,distance_label

#Gets the species k-mer vector, where the missing ones are filled with 0
def get_data(file_name,kmer_dir):
    data = {}
    AT =  list(pd.read_table('resource/k6.txt', header=None, index_col=0).T)
    for i in range(len(file_name)):
        m =pd.read_csv(kmer_dir+file_name[i]+'_k6.txt', header=None,sep='\t',index_col=[0]).T  
        #The missing kmer is supplemented with 0 
        
        new_data = (m.reindex(columns=AT, fill_value=0)).iloc[0]
        data[file_name[i]] =np.around((new_data/new_data.sum())*10000,4)     
    return data

#This weight is the w" in the paper
def weight_distance(weight,distance_label):
     new_label = []
     for i in distance_label:
         if i == 1:
             new_label.append(weight[i-1]*16)
         elif i == 2:
             new_label.append(weight[i-1]*8)
         elif i == 3:
             new_label.append(weight[i-1]*4)
         elif i == 4:
             new_label.append(weight[i-1]*2)
         elif i == 5:
             new_label.append(weight[i-1]*1)
     return new_label

'''
#This function saves the intermediate model
class callbackmodel(Callback):
    def __init__(self, model):
        self.model = model
    def on_epoch_end(self,epoch,logs=None):
        self.model.save('model/'+str(epoch)+'.h5')
'''

#triplet train
def train(kingdom,phylum,class_,order,family,genus,file_name,test_name,k,epochs_num,kmer_dir,output_dir):
    print('--------On_train_begin-----------')
       
    train_name = list(set(file_name).difference(set(test_name)))
    test_index = []
    train_index = []
    for i in test_name:
        test_index.append(file_name.index(i))
    for j in train_name:
        train_index.append(file_name.index(j))
   
    # train  
    anchor_list_train, positive_list_train,negative_list_train,distance_label = train_test_triplet(kingdom,phylum,class_,order,family,genus,file_name,train_index)                
    anchor_list_train, positive_list_train, negative_list_train,distance_label=shuffle(anchor_list_train,positive_list_train,negative_list_train,distance_label)

    #This weight is the w' in the paper
    weight = compute_class_weight('balanced', list(set(distance_label)), distance_label)    
    last_weight = weight_distance(weight,distance_label)       

   
    # test  
    anchor_list_test, positive_list_test,negative_list_test,nouse = train_test_triplet(kingdom,phylum,class_,order,family,genus,file_name,test_index)
   
    data = get_data(file_name,kmer_dir)
    anchor_data_train = list2data(anchor_list_train,data,k)
    positive_data_train = list2data(positive_list_train, data, k)
    negative_data_train = list2data(negative_list_train, data, k)

    anchor_data_test = list2data(anchor_list_test,data,k)
    positive_data_test = list2data(positive_list_test, data, k)
    negative_data_test = list2data(negative_list_test, data, k)
    
    model = triplet_model(k)
    model.fit([anchor_data_train,positive_data_train,negative_data_train],  np.zeros(len(anchor_data_train)), shuffle = False,
      epochs=epochs_num ,batch_size=5000,verbose =1,
      validation_data =([anchor_data_test,positive_data_test,negative_data_test],np.zeros(len(anchor_data_test))),
     # callbacks=[callbackmodel(model)],
      sample_weight = np.array(last_weight)
            )
    model.save(output_dir+'model.h5')

#File preparation
def preprocess_file(taxomony_csv,test_list_txt):
    train_data = pd.read_csv(taxomony_csv) 
     
    kingdom = list(train_data['kingdom'])
    phylum = list(train_data['phylum'])
    class_ = list(train_data['class'])
    order = list(train_data['order'])
    family = list(train_data['family'])
    genus = list(train_data['genus'])
    file_name = list(train_data.iloc[:, -1]) 

    test_name =  list(pd.read_table(test_list_txt, header=None, index_col=0).T)
    
    return kingdom,phylum,class_,order,family,genus,file_name,test_name
  
#Input parameters
if __name__ == "__main__":
    

    prog_base = os.path.split(sys.argv[0])[1]
    parser = optparse.OptionParser()

    parser.add_option("-i", "--inputcsv", action = "store", type = "string", dest = "genome_files_list",
                      help = "the taxomony of the input data")

    parser.add_option("-d", "--kmer_frequency_dir", action = "store", type = "string", dest = "kmer_frequency_dir",
                      help = "the dir of kmer frequency.")

    parser.add_option("-t", "--test_name", action = "store", type = "string", dest = "test_name",
                      help = "the list of test name.")
    
    parser.add_option("-k", "--kofKTuple", action = "store", type = "string", dest = "k_of_KTuple",
                      help = "the value k of KTuple")
    
    parser.add_option("-e", "--epochNum", action = "store", type = "string", dest = "epoch_num",default=50,
                      help = "the number of epoch")

    parser.add_option("-o", "--output", action = "store", type = "string", dest = "output_files",
                      help = "output dir")

    (options, args) = parser.parse_args()
    if (options.k_of_KTuple is None) or (options.epoch_num is None) or (options.output_files is None):
        print(prog_base + ": error: missing required command-line argument.")
        parser.print_help()
        sys.exit(0)
        
    if (options.genome_files_list is None) or (options.kmer_frequency_dir is None) or  (options.test_name is None):
        print(prog_base + ": error: missing required input command-line argument.")
        parser.print_help()
        sys.exit(0)      

    taxomony_csv = options.genome_files_list
    kmer_dir = options.kmer_frequency_dir
    test_list_txt = options.test_name

    k=int(options.k_of_KTuple)
    epochs_num = int(options.epoch_num)
    output_dir = options.output_files
  
    kingdom,phylum,class_,order,family,genus,file_name,test_name = preprocess_file(taxomony_csv,test_list_txt)

    train(kingdom,phylum,class_,order,family,genus,file_name,test_name,k,epochs_num,kmer_dir,output_dir)    


