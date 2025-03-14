"""
@author: Leon Scharwächter
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients

def sample_sphere(n, r):
    '''
    This function draws uniformly from an n-sphere such that the
    sample lies within the sphere.
    
    n = number of dimensions
    r = radius of the sphere

    '''
    # Generate random point on the unit n-sphere
    z = torch.randn(n).to(r.device)
    
    # Normalize to get a point on the unit sphere
    z /= torch.linalg.norm(z)
    
    # Sample a single radius
    # The volume of a sphere of radius r in n dimensions is proportional to r^n on [0,1],
    # that's why **(1/n) is used to ensure that the points are uniformly distributed within the volume
    # rather than being clustered near the origin
    
    #U = torch.rand(1) ** (1 / n)
    #U = U.to(r.device)
    #radius = U * r
    
    # Sample from Exponential distribution
    radius_distribution = torch.distributions.Exponential(rate=1.0)
    radius = radius_distribution.sample().clamp(max=r.item()).to(r.device)
    
    # Scale the unit vector by the radius
    perturbation = radius * z
    return perturbation

def sample_annulus(n, r_min, r_max):
    '''
    This function draws a sample from an n-dimensional annulus
    which is given by the two radii r_min and r_max.
    
    n = number of dimensions
    
    '''
    # Generate a random point on the unit n-sphere
    z = torch.randn(n).to(r_min.device)
    # Normalize to get a point on the unit sphere
    z /= torch.linalg.norm(z)  
    # Sample radius uniformly from the annular region
    U = torch.rand(1).to(r_min.device)
    r = torch.sqrt(U * (r_max**2 - r_min**2) + r_min**2)
    # Scale the unit vector by the radius
    perturbation = r * z
    return perturbation

def perturb_embeddings(embeddings: Tensor, tokenizer, scale=1, scale_neg=2):
    n_dim = embeddings.shape[-1] # embedding_dim
    
    if tokenizer.name_or_path == 'bert-base-uncased':
        std = torch.tensor(0.6359, device=embeddings.device)
    elif tokenizer.name_or_path == 'roberta-base':
        std = torch.tensor(1.6035, device=embeddings.device)
    
    # Initialize tensors
    perturbed_embeddings_pos = torch.zeros_like(embeddings)
    perturbed_embeddings_neg = torch.zeros_like(embeddings)

    # Iterate over every input of the batch
    for idx, embeds in enumerate(embeddings):
        # Iterate over every embedding of the input
        for e, embed in enumerate(embeds):                   
            # Perform positive perturbation
            noise_sphere = sample_sphere(n_dim, scale*std)
            perturbed_embeddings_pos[idx,e] = embed + noise_sphere
            
            # Perform negative perturbation
            noise_annulus = sample_annulus(n_dim, scale*std, scale_neg*std)
            perturbed_embeddings_neg[idx,e] = embed + noise_annulus

    return perturbed_embeddings_pos, perturbed_embeddings_neg

def restore_system_tokens(input_embeds_perturbed: Tensor, input_ids: Tensor, model, tokenizer):
    '''
    Undos the perturbation of system tokens (cls, pad, sep). 
    '''
    
    # extract embedding space
    if tokenizer.name_or_path == 'bert-base-uncased':
        embedding_matrix = model.pretrained.bert.embeddings.word_embeddings.weight.data
    elif tokenizer.name_or_path == 'roberta-base':
        embedding_matrix = model.pretrained.roberta.embeddings.word_embeddings.weight.data
    
    # initialize batch
    embeddings_restored = input_embeds_perturbed
    
    # Find system tokens in the original input and replace the perturbations
    # of the embeddings of these tokens with the unperturbed embedding vectors for each sequence in the batch
    for batch_idx, sequence in enumerate(input_ids):
        for token_pos, token_id in enumerate(sequence):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                # transform token into embedding
                embeddings_restored[batch_idx, token_pos] = embedding_matrix[token_id]

    return embeddings_restored


def convert_emb_to_ids(input_embeds_perturbed, input_ids, embedding_space, tokenizer, dist):
    '''
    Converts perturbed embeddings back into token IDs.
    Here, the sequences of perturbed embeddings are extracted from the batch and 
    for each sequence, every embedding emb is transformed to its nearest token in the embedding space.
    Thereby, this procedure ensures that every system token (CLS, PAD, SEP) remains untouched
    and that a perturbation does not result in a new system token.
    As long as input_ids contains the system token ids, this function also works with
    unrestored embeddings, where perturbations might have been done at system token embeddings.
    '''
    # Store the positions of system token ids for each sequence in the batch
    system_token_positions = [[] for _ in range(len(input_ids))]
    for batch_idx, sequence in enumerate(input_ids):
        for token_pos, token_id in enumerate(sequence):
            if token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
                system_token_positions[batch_idx].append(token_pos)
                
    # initialize batch
    input_ids_perturbed = []
    
    # for each perturbed embedding sequence in the batch...
    for batch_idx, perturbed_embeds in enumerate(input_embeds_perturbed):
        # find token ids and store them in a list
        perturbed_ids = []
        # for each embedded token in the sequence...
        for i, emb in enumerate(perturbed_embeds):
            # if current token is a system token, append it (and ignore perturbation)
            if i in system_token_positions[batch_idx]:
                perturbed_ids.append(input_ids[batch_idx,i])  
            else:
                if dist == 'euclidean':
                    # Find the nearest token id to the perturbed embedding (p2-norm)
                    nearest_token_ids = torch.argsort(torch.cdist(embedding_space.weight.data,
                                                                  emb.unsqueeze(0),
                                                                  p=2,
                                                                  compute_mode="use_mm_for_euclid_dist_if_necessary"),dim=0)
                elif dist == 'cosine':
                    # Find the token id with the highest cosine similarity to the perturbed embedding
                    nearest_token_ids = torch.argsort(F.cosine_similarity(embedding_space.weight.data, emb.unsqueeze(0), dim=1),
                                                      dim=0,descending=True)
                
                # Ensure that nearest token is not a CLS, PAD, or SEP token
                for token_id in nearest_token_ids:
                    if token_id.item() not in [tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id]:
                        perturbed_ids.append(token_id.item())
                        break  # Found valid token, exit loop
                else:
                    raise ValueError("No valid token found for the perturbed embedding when generating a counterexample.")           
        input_ids_perturbed.append(perturbed_ids)
    input_ids_perturbed = torch.tensor(input_ids_perturbed, device=input_ids.device)
    return input_ids_perturbed


class IntegratedGradientsBatch():
    '''
    to use for pretrained distilbert, switch model.pretrained.bert.embeddings
    to model.pretrained.distilbert.embeddings twice
    and comment out token_type_ids in additional_forward_args=()
    '''
    def __init__(self, model, tokenizer, device):
        # Initialize (Layer) Integrated Gradients
        if tokenizer.name_or_path == 'bert-base-uncased':
            self.lig = LayerIntegratedGradients(model, model.pretrained.bert.embeddings)
            self.embedding_matrix = model.pretrained.bert.embeddings.word_embeddings.weight.data
        elif tokenizer.name_or_path == 'roberta-base':
            self.lig = LayerIntegratedGradients(model, model.pretrained.roberta.embeddings)
            self.embedding_matrix = model.pretrained.roberta.embeddings.word_embeddings.weight.data
        self.cls_token_id = tokenizer.cls_token_id # Token to begin with
        self.pad_token_id = tokenizer.pad_token_id # Token used for padding
        self.sep_token_id = tokenizer.sep_token_id # Token added to the end of the text
        self.device = device
        
    def forward(self, input_ids, token_type_ids, attention_mask, predicted_classes, n_steps):
        # Create an individual baseline for each input sequence
        baselines = torch.tensor([], dtype=torch.int64, device=self.device)
        for _, ids in enumerate(input_ids):
            sep_position = (ids == self.sep_token_id).nonzero().squeeze().item() # find SEP-Token
            baseline = [self.pad_token_id] * len(ids) # initialize baseline with PAD-Tokens
            baseline[sep_position] = self.sep_token_id # Place SEP-Token at respective position
            baseline[0] = self.cls_token_id # Start baseline with CLS-Token
            baseline = torch.tensor([baseline], device=self.device)
            baselines = torch.cat((baselines, baseline),dim=0) # Append baseline
            
        # As input ids are fed into the model and not embedding vectors,
        # use_embeds is set to False (as the attributions should be calculated
        # on the token-level)
        # use_softmax ensures that the output of the model are probabilities
        # and not a sequence of logits and hidden states
        use_embeds=False
        use_softmax=True

        # Compute attributions using Integrated Gradients
        attributions, delta = self.lig.attribute(inputs=input_ids,
                                                 baselines=baselines,
                                                 target=predicted_classes,
                                                 additional_forward_args=(token_type_ids,
                                                                          attention_mask, 
                                                                          use_embeds, use_softmax),
                                                 n_steps=n_steps,
                                                 return_convergence_delta=True)
        
        # In this implementation, the overall attribution score is the 
        # sum of the attributions per hidden dimension
        attributions = attributions.sum(dim=-1)
        return attributions
    
    def forward_embeds(self, input_embeds,
                       token_type_ids,
                       attention_mask, predicted_classes, n_steps):
        cls_emb = self.embedding_matrix[self.cls_token_id]
        sep_emb = self.embedding_matrix[self.sep_token_id]
        pad_emb = self.embedding_matrix[self.pad_token_id]
        baselines = torch.tensor([], dtype=torch.float32, device=self.device)
        for _, embeds in enumerate(input_embeds):
            sep_position = torch.nonzero((embeds==sep_emb).all(dim=1)).squeeze().item() # find SEP-Embedding index
            baseline = [pad_emb] * len(embeds) # initialize baseline with PAD-Embeddings
            baseline[sep_position] = sep_emb # Place SEP-Embedding at respective position
            baseline[0] = cls_emb # Start baseline with CLS-Embedding
            baseline = torch.stack(baseline).to(self.device)
            #baseline = torch.tensor(baseline, device=self.device)
            baselines = torch.cat((baselines, baseline.unsqueeze(0)),dim=0) # Append baseline
        
        # Specify input arguments
        # (ig.attribute() needs every input argument to be specified)
        #token_type_ids = attention_mask = []
        use_embeds=True
        use_softmax=True
        
        # Compute attributions using Integrated Gradients
        attributions, delta = self.lig.attribute(inputs=input_embeds,
                                                 baselines=baselines,
                                                 target=predicted_classes,
                                                 additional_forward_args=(token_type_ids,
                                                                          attention_mask, 
                                                                          use_embeds, use_softmax),
                                                 n_steps=n_steps,
                                                 return_convergence_delta=True)
        
        # In this implementation, the overall attribution score is the 
        # sum of the attributions per hidden dimension
        attributions = attributions.sum(dim=-1)
        return attributions
    

def extract_top_k_attributions(tokens: str, attributions: float, k: int, return_tensors=False):
    # initialize batch of top tokens and attributions
    top_k_tokens_batch = []
    top_k_attributions_batch = []
    
    # iterate over the batch of token sequences
    for batch_idx, sequence in enumerate(tokens):
        # Sort attributions array in descending order
        sorted_indices = sorted(range(len(attributions[batch_idx])), key=lambda i: attributions[batch_idx,i], reverse=True)
    
        # Extract top k attributions and corresponding tokens
        top_k_tokens = [tokens[batch_idx,i] for i in sorted_indices[:k]]
        top_k_attributions = [attributions[batch_idx,i] for i in sorted_indices[:k]]
        
        # Append to batch
        top_k_tokens_batch.append(top_k_tokens)
        top_k_attributions_batch.append(top_k_attributions)
        
    if return_tensors:
        top_k_tokens_batch = torch.tensor([[[embedding for embedding in sequence] for sequence in batch] for batch in top_k_tokens_batch])
        top_k_attributions_batch = torch.tensor(top_k_attributions_batch)
        
    return top_k_tokens_batch, top_k_attributions_batch

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.
    
    Parameters:
    p (torch.Tensor): Target probability distribution (logits or probabilities).
    q (torch.Tensor): Predicted probability distribution (logits or probabilities).
    
    Returns:
    torch.Tensor: The KL divergence loss.
    """
    # p and q need to be normalized, e.g. through a softmax function
    # Ensure the distributions are normalized (i.e., they sum to 1)
    #p = F.softmax(p, dim=1)
    #q = F.softmax(q, dim=1)
    
    # Compute the KL Divergence
    kl_div = F.kl_div(q.log(), p, reduction='batchmean')
    
    return kl_div

class JSDivergence(nn.Module):
    '''
    provides a smoothed and normalized version of KL divergence,
    with scores between 0 (identical) and 1 (maximally different),
    when using the base-2 logarithm.
    '''
    def __init__(self,
                 reduction='batchmean', eps=1e-8):
        super(JSDivergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.eps = eps  # small const to avoid NaNs from log(0)

    def forward(self, p: torch.tensor, q: torch.tensor):
        #p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        p = p + self.eps
        q = q + self.eps
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

# from https://medium.com/@babux1
def top_k_frequent(nums: list[int], k: int) -> list[int]:
        # Create a hash table to store the frequency of each element
        freq = {}
        for num in nums:
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        # Sort the hash table by frequency in descending order
        freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
        
        # Select the k elements with the highest frequency
        result = []
        i = 0
        for num, count in freq.items():
            result.append(num)
            i += 1
            if i == k:
                break
        
        return result

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
        
    def forward(self, probs):
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=1) 
        # Mean entropy over the batch (reduction=mean)
        mean_entropy = torch.mean(entropy)
        return -mean_entropy # Return negative to promote entropy maximization
    
class LogitMarginLoss(nn.Module):
    def __init__(self, margin=0.5, reduction='mean'):
        """
        Args:
            margin (float): The margin by which the original class logit should be lower than the highest logit of the other classes.
            reduction (str): Specifies the reduction to apply to the output. 
                             Must be 'none', 'mean' or 'sum'.
        """
        super(LogitMarginLoss, self).__init__()
        self.margin = margin
        assert reduction in ['none', 'mean', 'sum'], "Reduction must be one of 'none', 'mean', or 'sum'"
        self.reduction = reduction

    def forward(self, negative_logits, original_class):
        """
        Args:
            negative_logits (torch.Tensor): Logits predicted by the model for negative perturbations (batch_size x num_classes).
            original_class (torch.Tensor): The ground truth class of the original input (batch_size).
            
        Returns:
            torch.Tensor: The logit margin loss for the negative perturbations.
        """
        batch_size, num_classes = negative_logits.size()

        # Gather the logit for the original class (for each sample in the batch)
        logits_of_truelabel = negative_logits.gather(1, original_class.view(-1, 1)).squeeze(1)

        # Get the max logit for all classes except the original class
        mask = torch.ones_like(negative_logits).bool()
        mask.scatter_(1, original_class.view(-1, 1), False)  # Mask out the original class
        logits_maxother = negative_logits.masked_select(mask).view(batch_size, num_classes - 1).max(dim=1)[0]

        # Calculate the logit margin loss
        loss = torch.clamp(logits_of_truelabel - logits_maxother + self.margin, min=0) # clamp similar to max here

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def intersection_proportion(list1: list, list2: list):
    set1 = set(list1)
    set2 = set(list2)
    overlap = set1 & set2 
    if len(set1) == len(set2):
        intersection_prop = float(len(overlap))/len(set1)
        return intersection_prop
    else:
        raise ValueError("Both lists of top attributions should have the same length.")
        
def top_k_to_attention(attributions: Tensor, k):
    '''
    This function takes as input argument an attribution scores tensor of shape [bz, len] and
    returns an attention mask of the same shape, where the attention is set to 1 for the
    k top highest (positive) attribution scores and the rest is set to 0.
    If the number of positive scores is smaller than k, the number of attention values = 1
    equals the number of positive scores.
    '''
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions)
        
    attention_mask = torch.zeros_like(attributions)
    # Iterate over the batch dimension
    for i in range(attributions.shape[0]):
        # Get the top-k indices per input
        top_k_indices = torch.topk(attributions[i], k).indices
        # Filter out indices where the attribution is not greater than 0
        valid_top_k_indices = top_k_indices[attributions[i][top_k_indices] > 0]
        # Set the valid top-k positions to 1 in the attention tensor
        attention_mask[i, valid_top_k_indices] = 1 
    return attention_mask

def cosine_distance(x1, x2, dim=1):
    # use dim=1 when working with shape [bach_size, embedding_size] (e.g. CLS-tokens as sentence embedding)
    # use dim=2 when working with embeddings [batch_size, input_size, embedding_size] (e.g. single token embeddings)
    cos_sim = F.cosine_similarity(x1, x2, dim=dim)
    return 1 - cos_sim  # Cosine distance is 1 - cosine similarity, output range: [0 - 2]

def pearson_correlation_distance(x_batch: Tensor, y_batch: Tensor) -> Tensor:
    # Ensure both batches have the same shape
    assert x_batch.shape == y_batch.shape, "Input tensors must have the same shape"
    
    # Center the data (subtract the mean along the sequence dimension)
    x_batch_centered = x_batch - x_batch.mean(dim=1, keepdim=True)
    y_batch_centered = y_batch - y_batch.mean(dim=1, keepdim=True)
    
    # Compute covariance
    covariance = torch.sum(x_batch_centered * y_batch_centered, dim=1) / (x_batch.size(1) - 1)
    
    # Compute standard deviations
    std_x = torch.std(x_batch, dim=1)
    std_y = torch.std(y_batch, dim=1)
    
    # Compute Pearson correlation coefficient
    correlation = covariance / (std_x * std_y)
    
    # Transform into a distance
    corr_dist = 1 - ((1 + correlation) / 2)
    
    # output range: [0 - 1]
    return corr_dist

def mean(lst: list):
    '''
    Calculates the mean of a list on cuda without being a tensor or numpy object.
    '''
    return sum(lst) / len(lst) 

def mask_special_chars(tokenizer, input_ids, attention_mask):
    # -- NOT RECOMMENDED TO USE -- ONLY FOR TESTING PURPOSES --
    # Iterate over input_ids and set the attention to 0 where
    # a special char occurs. Returns the adapted attention mask
    # that can be used to feed into the model.
    special_chars = ['.', ',', '!', '?', "'", '"', '-',"(",")","[","]",
                     '/',"\\",'§','$','%','&','=','@','#','*','+','_',
                     '<','>','|','^','°','~','{','}',":",";"]   
    special_chars_ids = tokenizer.convert_tokens_to_ids(special_chars)
    special_chars_ids = torch.tensor(special_chars_ids, device=input_ids.device)
    
    for i in range(input_ids.shape[0]):  # batch size
        for j in range(input_ids.shape[1]):  # max length
            if input_ids[i, j].item() in special_chars_ids:
                attention_mask[i, j] = 0
    
    return attention_mask

def count_ids(input_ids: Tensor, tokenizer):
    # Store system tokens
    system_tokens = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],device=input_ids.device)
    # Create a mask where system tokens are marked as True
    system_tokens_mask = torch.isin(input_ids, system_tokens)
    # Invert the mask to get non-system tokens
    non_system_tokens_mask = ~system_tokens_mask
    # Count and return the number of non-system tokens
    return non_system_tokens_mask.sum().item()

def filter_embeddings_by_attention(embeddings, attention):
    """
    Filter embeddings based on the attention vector. This function only works with single examples, i.e.
    where the batch size == 1.
    
    Args:
        embeddings (torch.Tensor): The input embeddings of shape [1, seq_len, embedding_size]
        attention (torch.Tensor): The attention vector of shape [1, seq_len]

    Returns:
        torch.Tensor: A new tensor containing the filtered embeddings
    """
    # Check that the input tensors have the expected dimensions
    assert len(embeddings.shape) == 3 and embeddings.shape[0] == 1, "Embeddings should be of shape [1, seq_len, embedding_size]"
    assert len(attention.shape) == 2 and attention.shape[0] == 1, "Attention should be of shape [1, seq_len]"

    # Drop the batch dimension to make indexing easier
    embeddings = embeddings.squeeze(0)  # [seq_len, embedding_size]
    attention = attention.squeeze(0)    # [seq_len]

    # Get the indices where attention is 1
    indices = torch.nonzero(attention == 1).squeeze(1) 

    # Use the indices to select the corresponding embeddings
    filtered_embeddings = embeddings[indices]  # [num_selected, embedding_size]

    return filtered_embeddings

def count_differing_ids(tensor1, tensor2):
    """
    Takes two PyTorch tensors of the same shape and returns the count of differing indices for each batch.
    
    Args:
    - tensor1: A PyTorch tensor of shape (batch_size, seq_length).
    - tensor2: A PyTorch tensor of shape (batch_size, seq_length).
    
    Returns:
    - A PyTorch tensor of shape (batch_size,), where each value indicates the count of differing indices for that batch.
    """
    # Ensure that the tensors have the same shape
    assert tensor1.shape == tensor2.shape, "The input tensors must have the same shape."

    # Calculate the differences by comparing corresponding elements
    differing_indices = tensor1 != tensor2
    
    # Sum the number of differing indices along the sequence length for each batch
    differences_per_batch = differing_indices.sum(dim=1)
    
    return differences_per_batch