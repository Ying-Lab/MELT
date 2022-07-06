import torch
import train
import numpy as np


if __name__ == "__main__":
     
    # device = torch.device('cuda:1')
    device = 'cpu'

    with torch.no_grad():
    
        remove_list = train.remove_list
        dataset_path = train.dataset_path
        trained_model_path = 'trained_model.pth'
        
        #get data
        all_data,all_label=train.get_data_otu(file_name=dataset_path,remove=[])
        all_data = torch.tensor(all_data,device=device)
        
        #load the trained model
        new_model = train.TNN(all_data.shape[1]).to(device)
        new_model.load_state_dict(torch.load(trained_model_path,map_location='cpu'))
        new_model.eval()

        # predict 
        # the output test embedding shape is (len(remove_list),128) 
        all_data_embedding= new_model(all_data.float()).cpu()
        test_data_embedding = all_data_embedding[remove_list]

        # save the embedding to csv file
        np.savetxt('all_data_embedding.csv',all_data_embedding,delimiter=',')
        np.savetxt('test_data_embedding.csv',test_data_embedding,delimiter=',')