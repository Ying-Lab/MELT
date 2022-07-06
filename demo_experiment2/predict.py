import torch
import train

import numpy as np

def get_data():

    X_train,cell_train_label= train.makedata(train_dataset_path,shuffle_flag=False)
    X_test,cell_test_label= train.makedata(test_dataset_path,shuffle_flag=False)
    return X_train,cell_train_label,X_test,cell_test_label



if __name__ == "__main__":

    train_dataset_path = 'data/train.csv'
    test_dataset_path = 'data/test.csv'
    trained_model_path = 'trained_model.pth'
    # device = torch.device('cuda:0')
    device = 'cpu'
    with torch.no_grad():
        X_train,cell_train_label,X_test,cell_test_label = get_data()
        
        # load trained model 
        new_model = train.TNN(X_train.shape[1]).to(device)
        new_model.load_state_dict(torch.load(trained_model_path,map_location='cpu'))  
        
        new_model.eval()
        X_train = torch.tensor(X_train,device=device)
        X_test = torch.tensor(X_test,device=device)

        # predict embedding
        train_embedding= new_model(X_train.float())
        test_embedding = new_model(X_test.float())

        train_embedding=train_embedding.cpu()
        test_embedding=test_embedding.cpu()

        # save embedding to csv file
        np.savetxt('test_embedding.csv',test_embedding,delimiter=',')
        np.savetxt('train_embedding.csv',train_embedding,delimiter=',')


        