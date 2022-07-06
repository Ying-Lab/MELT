# -*- coding: utf-8 -*-
import os
#if you dont want to use CPU,you can delete it.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from tensorflow.compat.v1.keras.models import Model,load_model
from tensorflow.compat.v1.keras import backend as K
from collections import defaultdict
import sys
import optparse


#convert species names into corresponding k-mer vectors
def list2data(list_file, kmer_data, k):
    file_number = len(list_file)
    data = np.zeros((file_number, 4**k),dtype='float32')
    for i in range(len(list_file)):
        data[i] = (kmer_data[list_file[i]])
    return data

# compute loss, x is anchor，y is positive, z is negtive
def eulidean_distance(vects):
    x,y,z = vects
    sum_square = K.sqrt(K.sum(K.square(x-y),axis=1,keepdims=True))-K.sqrt(K.sum(K.square(x-z),axis=1,keepdims=True))+0.5
   # sum_square = K.sqrt(K.sum(K.square(x-y),axis=1,keepdims=True))-K.sqrt(K.sum(K.square(x-z),axis=1,keepdims=True))
    return K.maximum(sum_square,0)

#This is the model output value because there is no y_label
def identity_loss(y_true,y_pred):   
    return K.mean(y_pred-0*y_true)

def list_duplicates(seq):    
    tally = defaultdict(list)    
    for i,item in enumerate(seq):        
        tally[item].append(i)    
    return ((key,locs) for key,locs in tally.items() if len(locs)>=1) 

#Gets the species k-mer vector, where the missing ones are filled with 0
def get_data(file_name,kmer_dir):
    data = {}
    AT = list(pd.read_table('resource/k6.txt', header=None, index_col=0).T)
    for i in range(len(file_name)):
         m =pd.read_csv(kmer_dir+file_name[i]+'_k6.txt', header=None,sep='\t',index_col=[0]).T
         new_data = (m.reindex(columns=AT, fill_value=0)).iloc[0]
     #   data[file_name[i]] =np.around((new_data)*10000,6)
         data[file_name[i]] =np.around((new_data/new_data.sum())*10000,6)
    return data

#File preparation abd position
if __name__ == "__main__":
    prog_base = os.path.split(sys.argv[0])[1]
    parser = optparse.OptionParser()

    parser.add_option("-i", "--inputcsv", action = "store", type = "string", dest = "genome_files_list",
                      help = "the taxomony of the input data")

    parser.add_option("-d", "--kmer_frequency_dir", action = "store", type = "string", dest = "kmer_frequency_dir",
                      help = "the dir of kmer frequency.")

#    parser.add_option("-k", "--kofKTuple", action = "store", type = "string", dest = "k_of_KTuple",
 #                     help = "the value k of KTuple")

    parser.add_option("-t", "--test_name", action = "store", type = "string", dest = "test_name",
                      help = "the list of test name.")

 
    parser.add_option("-o", "--output", action = "store", type = "string", dest = "output_files",
                      help = "output dir")

    (options, args) = parser.parse_args()
    if (options.output_files is None):
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

#    k=int(options.k_of_KTuple)
 #   epochs_num = int(options.epoch_num)
    output_dir = options.output_files


    train_data = pd.read_csv(taxomony_csv)
    all_name_ = list(train_data.iloc[:, -1])

    test_name = list(pd.read_table(test_list_txt,header = None,index_col=0).T)
    train_name = list(set(all_name_).difference(set(test_name)))
    phylum_ = list(train_data['phylum'])
    class_ = list(train_data['class'])
    order_ = list(train_data['order'])
    family_ = list(train_data['family'])
    genus_ = list(train_data['genus'])
    data = get_data(all_name_,kmer_dir)

    test_category = {}
    for test_iterm in test_name:
        test_category[test_iterm] = genus_[all_name_.index(test_iterm)]

    genus = []
    all_name = []
    cc = list(zip(genus_,all_name_))
    mid = []
    for test_iterm in train_name:
        mid.append(cc[all_name_.index(test_iterm)])
    genus[:],all_name[:]= zip(*mid)   
    non_unique_train = []
    for dup in sorted(list_duplicates(genus)):
        non_unique_train.append(dup)
    
    model1 = load_model(output_dir+'model.h5',custom_objects={'identity_loss':identity_loss,'eulidean_distance':eulidean_distance})
    model = Model(inputs=model1.input[0],outputs=model1.get_layer('Dense_20').get_output_at(0))
    g = 0
    f = 0
    o = 0
    c =0
    p = 0
    n = 0
    file = open(output_dir+'predict_taxonomy.txt','w')
    file.write('predict_phylum'+'\t'+'predict_class'+'\t'+'predict_order'+'\t'+'predict_family'+'\t'+'predict_genus'+'\t'+'species'+'\n')
    for anchor_spiece in test_name:
        all_ = {}
        for positive_iterm in non_unique_train: 
       
            name = positive_iterm[0]
            test_genus = [anchor_spiece]
            for i in positive_iterm[1]:
                test_genus.append(all_name[i])
            input_data = list2data(test_genus,data,6)
            output_data = model.predict(input_data)
            oushi = []
            for j in output_data[1:]:
                oushi.append(np.sqrt(sum(np.power((output_data[0] - j), 2))))           
            all_[name] = np.mean(oushi)
                   
        all_ = sorted(all_.items(), key=lambda x: x[1])
        file.write(phylum_[genus_.index(all_[0][0])]+'\t'+class_[genus_.index(all_[0][0])]+'\t'+order_[genus_.index(all_[0][0])]+'\t'+family_[genus_.index(all_[0][0])]+'\t'+all_[0][0]+'\t'+anchor_spiece+'\n')
    file.close()


              
                    
    
