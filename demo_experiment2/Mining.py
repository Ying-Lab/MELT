import os
import random
from itertools import permutations
import numpy as np
import torch




def _get_triplet_mask(labels, device='cpu'):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size()[0]).type(torch.ByteTensor) # make a index in Bool 
    indices_not_equal = -(indices_equal-1) # flip booleans
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k  

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = (i_equal_j & ~i_equal_k)
    # Combine the two masks
    mask_main = (distinct_indices.to(device).bool() & valid_labels)

    return mask_main


def _pairwise_distances(embeddings,dist_model, squared=False, device='cpu'):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        dist_model: choose which distance to ues , 'cos' represent cosine distance ,  'euclidean distances'
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    embeddings = torch.squeeze(embeddings,1)  # shape=(batch_size, features, 1)
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings,0,1)) # shape=(batch_size, batch_size)
    
    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    if(dist_model=='euclidean distances'):
        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)
        
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = torch.max(distances, torch.Tensor([0.0]).to(device))
        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)
        # return distances matrix (batch X batch)
        return distances

    if(dist_model=='cos'): 
        # Compute the pairwise distance matrix as we have:
        # cos<a,b>= (a·b)  /  ( ||a|| * ||b|| )
        square_norm_matrix1 = torch.unsqueeze(square_norm,1) * torch.ones(square_norm.shape[0],square_norm.shape[0]).to(device)
        square_norm_matrix2 = torch.unsqueeze(square_norm,0) * torch.ones(square_norm.shape[0],square_norm.shape[0]).to(device)
        
        square_norm_matrix = square_norm_matrix1 * square_norm_matrix2
        
        mask = (square_norm_matrix == 0.0).float()
        square_norm_matrix = square_norm_matrix + mask * (1e-16)  
        
        norm_matrix = torch.sqrt(square_norm_matrix)
        
        norm_matrix = norm_matrix * (1.0-mask)
        
        cos_similarity = dot_product / norm_matrix
        distances = 1.0 - cos_similarity

        return distances





def online_mine_all(labels, embeddings, margin, dist_model, squared=False, device='cpu'):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        dist_model : choose which distance to ues , 'cos' represent cosine distance ,  'euclidean distances'
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        num_positive_triplets: the number of positive triplets (where triplet_loss > 0) ,
                            this number will change dynamically during tarining
        num_positive_triplets_no_margin: Similar to num_positive_triplets ,but do not thinkabout margin when compute triplet loss
        num_valid_triplets: the number of valid triplets (where label(a) == label(p) and label(a) != label(n)),
                            this number will not change and num_positive_triplets <= num_valid_triplets
        torch.max(pairwise_dist): the max distance between two embedding
    """
    # Get the pairwise distance matrix,(batch_size,batch_size)
    pairwise_dist = _pairwise_distances(embeddings, dist_model=dist_model ,squared=squared, device=device)
    # the shape is (batch_size, batch_size, 1) after expand dimension
    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2) 
    #assert
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    # triplet_loss_no_margin is a 'triplet_loss' without margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    triplet_loss_no_margin = anchor_positive_dist - anchor_negative_dist 

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels, device=device)
    mask = mask.float()
    triplet_loss = mask*triplet_loss
    triplet_loss_no_margin = mask*triplet_loss_no_margin

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.max(triplet_loss, torch.Tensor([0.0]).to(device))
    triplet_loss_no_margin =  torch.max(triplet_loss_no_margin, torch.Tensor([0.0]).to(device))

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = (triplet_loss > 1e-16).float()
    valid_triplets_no_margin = (triplet_loss_no_margin > 1e-16).float()
    
    num_positive_triplets = torch.sum(valid_triplets)
    num_positive_triplets_no_margin = torch.sum(valid_triplets_no_margin)
    
    # Count number of valid triplets (where label(a) == label(p) and label(a) != label(n))
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    
    return triplet_loss, num_positive_triplets,num_positive_triplets_no_margin, num_valid_triplets,torch.max(pairwise_dist)



