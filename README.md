# MELT
#### **ME**tric **L**earning with **T**riplet network for Comparing Genomic data

Comparison between genomic data is critical to estimating evolutionary relationships, understanding the microbiome  dynamics, or exploring the developmental process of embryogenesis.

Alignment-based and alignment-free methods play an important role is genomic comparison. However, the current existing measures are mostly based on a fixed dissimilarity function or model, which might not be suitable for all scenarios. 

Therefore, in this study, we proposed MELT, a metric learning model with triplet network to obtain a suitable dissimilarity measure for hierarchical or longitudinal genomic and microbiome  data. The construction of the training triplet dataset sufficiently reflects the dissimilarity characteristics of application scenarios. Hence, MELT offers a data-based, scenario-oriented framework for adaptive metric comparison.

#### Compared to existing comparison methods, MELT offers the following advantages:
- MELT learns useful representations of genomic data by the reference relationship that “data A is closer to data B than to data C ”, which sufficiently reflects the dissimilarity characteristics of application scenarios
- Instead of accurate alignment distances for training, MELT requires only dissimilarity comparisons among A, B, and C. The embedding function is learned automatically from the dissimilarity comparisons. Therefore, MELT is particularly suitable to deal with scenarios without clear categorical information, such as hierarchical or longitudinal datasets.

The experiments on comparing genomic sequences, temporal microbiome  samples, and gene expression profiles  from scRNA-seq demonstrate the significant performance in the three application scenarios. 


## Environment/Package installation 
### We offer two methods

- ### Install via conda-pack file(recommended)
**Conda-pack** is a command line tool for creating archives of conda environments that can be installed on other systems and locations.

Here we use **conda-pack** to pack our environment and provide it on release. You can easily download these environments to run our code without pre-installing anaconda or python.

- Make sure your operating system is Linux
- In the terminal window of the Linux operating system
- Detailed steps
1. Download the source code to your directory. 
```
$git clone https://github.com/Ying-Lab/MELT.git
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/github1.gif)
2. Enter the MELT directory: 

```
$cd MELT/
```
3. Download the environment .tar.gz file. This will take some time depending on your internet.

```
$wget https://github.com/Ying-Lab/MELT/releases/download/v1.0/MELT-pytorch-env.tar.gz

$wget https://github.com/Ying-Lab/MELT/releases/download/v1.0/MELT-tensorflow-env.tar.gz
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/GitHub2.gif)
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/GitHub3.gif)
4. Extract the environment .tar.gz file to a specific directory

```
$mkdir MELT-pytorch-env
$tar -zxvf MELT-pytorch-env.tar.gz -C MELT-pytorch-env
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/GitHub4.gif)
```
$mkdir MELT-tensorflow-env
$tar -zxvf MELT-tensorflow-env.tar.gz -C MELT-tensorflow-env
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/GitHub5.gif)
5.You have now successfully installed the environment. You can use the following command to activate an environment.
- activate pytorch env
```
$source MELT-pytorch-env/bin/activate
```
- activate tensorflow env

```
$source MELT-tensorflow-env/bin/activate
```
- ### Installation via Anaconda (not recommended)
If you find conda-pack files difficult to download, you can choose this method to install the environment.
- In the terminal window of the Linux operating system
- Detailed steps:
1. Install Miniconda to manage your environment (you can skip this step if you already have Conda installed). 
```
$wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$bash Miniconda3-latest-Linux-x86_64.sh -bfp
```
2. Restart your terminal
3. Download the source code to your directory. 
```
$git clone https://github.com/Ying-Lab/MELT.git
```
4. Create an environment and install the package from **.yml** file. Conda will download th package. 
    The new environment name is 'MELT-pytorch-env'

```
$cd MELT/
$conda env create -f MELT-pytorch-env.yml
$conda env create -f MELT-tensorflow-env.yml
```
5. Activate  environment
- activate pytorch env

```
$conda activate MELT-pytorch-env
```
- activate tensorflow env

```
$conda activate MELT-tensorflow-env
```
Now you've successfully installed the environment 

## The trained models of MELT
- We provided three trained models for each of the three experiments:
	- Experiment1: MELT/demo_experiment1/resource/trained_model.h5
	- Experiment2: MELT/demo_experiment2/trained_model.pth
	- Experiment3: MELT/demo_experiment3/trained_model.pth


## The demo of MELT

Three small datasets are used for demonstration purposes. 

### MELT for hierarchical relationships of genomic sequences
Please activate tensorflow-env first
```
$source MELT-tensorflow-env/bin/activate
```
or
```
$conda activate MELT-tensorflow-env
```

1. Usage of MELT

- The main running commands are as follows:

     -h, --help: show help information
     
     -i, --inputcsv: the taxomony of the input data
     
     -d, --kmer_frequency_dir: the dir of kmer frequency.
     
     -t, --test_name: the list of test name.
     
     -k, --kofKTuple: the value k of KTuple
     
     -e, --epochNum: the number of epoch.
     
     -o, --output: output dir.

2. Run MELT to get model.

Extract the zip file:    
```
$unzip demo_experiment1/resource/kmer.zip -d demo_experiment1/resource/
```
    
Create a new folder to put model file:

```   
$mkdir demo_experiment1/output
```  
Enter demo_experiment1 directory:
    
```
$cd demo_experiment1/
```

Run triplet_model.py:
```  
$python code/triplet_model.py -i resource/data.csv -d resource/kmer/ -t resource/test_name.txt -k 6 -e 30 -o output/
```  
 
3. Predict taxonomy of unknown species.

Run taxonomy_localization.py
    
```   
$python code/taxonomy_localization.py -i resource/data.csv -d resource/kmer/ -t resource/test_name.txt  -o output/
``` 
The output are ./output/predict_taxonomy.txt.

Here we also provide a trained model in resource/trained_model.h5 

4. back to MELT/ path

```
$cd ..
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/MELT%20for%20hierarchical%20relationships%20of%20genomic%20sequences.gif)
### MELT for longitudinal cell division on scRNA-seq data
Please activate pytorch-env first
```
$source MELT-pytorch-env/bin/activate
```
or
```
$conda activate MELT-pytorch-env
```
1. Enter demo_experiment2 directory

```
$cd demo_experiment2/
```
    
2. Run MELT to train the model. 
    It will save every epoch model during training(default) in ./model/ directory.
```
$python ./train.py
```
3. Predict embedding of test data. Here we provide a trained model in ./trained_model.pth .
It will produce two csv files in the current path : 
- train_embedding.csv 
- test_embedding.csv
```
$python ./predict.py
```
4. back to MELT/ path

```
$cd ..
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/MELT%20for%20longitudinal%20cell%20division%20on%20scRNA-seq%20data.gif)
### MELT for longitudinal dissimilarity of temporal microbiome data
Please activate pytorch-env first
```
$source MELT-pytorch-env/bin/activate
```
or
```
$conda activate MELT-pytorch-env
```
1.  Enter experiment3 directory

```
$cd ./demo_experiment3
```
    
2. Run MELT to train the model. 
    It will save every epoch model during training(default) in ./model/ directory.
```
$python ./train.py
```
3. Predict embedding of test data. Here we provide a trained model in ./trained_model.pth .
It will produce two csv files in the current path :
- all_data_embedding.csv 
- test_data_embedding.csv
```
$python ./predict.py
```
![image](https://github.com/Ying-Lab/MELT/blob/main/gif/MELT%20for%20longitudinal%20dissimilarity%20of%20temporal%20microbiome%20data.gif)
