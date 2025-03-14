"""
@author: Leon Scharwächter
"""

import torch
from transformers import AutoTokenizer
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import TensorDataset, DataLoader

#%%

def getDataloader(MODEL_NAME: str, BATCH_SIZE: int, subset: str = 'train'):
    # Load the dataset
    newsgroups_data = fetch_20newsgroups(subset=subset,
                                         remove=('headers', 'footers', 'quotes'))

    # Tokenize and pad the input sequences
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = []
    token_type_ids = []
    attention_masks = []

    labels = torch.tensor(newsgroups_data.target)  # Load the labels
    num_labels = len(labels.unique())

    for text in newsgroups_data.data:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True, # Add [CLS]- and [SEP]-Tokens
                            max_length = tokenizer.model_max_length,
                            padding = 'max_length',
                            truncation=True,
                            return_token_type_ids = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        
        # Add the encoded sentence to the list 
        input_ids.append(encoded_dict['input_ids'])
        
        # Add encoded token type index to the list 
        token_type_ids.append(encoded_dict['token_type_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)

    # Create a TensorDataset
    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, labels)

    # Create a DataLoader with batch size of BATCH_SIZE
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            pin_memory=True)
    
    return dataloader, labels, num_labels

def getDataset(subset: str = 'train'):  
    # Load the dataset
    newsgroups_data = fetch_20newsgroups(subset=subset,
                                          remove=('headers', 'footers', 'quotes'))
    # Extract labels
    labels = torch.tensor(newsgroups_data.target) 
    return newsgroups_data, labels

'''
newsgroups_data.target_names

 0['alt.atheism',
 1'comp.graphics',
 2'comp.os.ms-windows.misc',
 3'comp.sys.ibm.pc.hardware',
 4 'comp.sys.mac.hardware',
 5'comp.windows.x',
 6'misc.forsale',
 7'rec.autos',
 8'rec.motorcycles',
 9'rec.sport.baseball',
10'rec.sport.hockey',
11'sci.crypt',
12'sci.electronics',
13'sci.med',
14'sci.space',
15'soc.religion.christian',
16'talk.politics.guns',
17'talk.politics.mideast',
18'talk.politics.misc',
19'talk.religion.misc']
'''

#%%

def extract_inputs_by_label(dataset, target_name: str, amount: int = None):
    # Gets examples from the test dataset...
    
    # Lists to store input data
    input_texts = []

    # Iterate through the dataset
    for i, text in enumerate(dataset.data):
        label = dataset.target_names[dataset.target[i]]
        if label == target_name:
            input_texts.append(text)
            
    if amount is None:
        return input_texts
    else:
        return input_texts[:amount]

def tokenize_one_example(tokenizer, input_text: str):

    # Tokenize input text and convert to tensor
    inputs = tokenizer.encode_plus(input_text,
                                   #inputs_filtered[example_nr],
                                   add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                   max_length = tokenizer.model_max_length, 
                                   padding = 'max_length',
                                   truncation=True,
                                   return_token_type_ids = True,
                                   return_attention_mask = True,   
                                   return_tensors = 'pt'
                                   )
    
    input_ids = inputs["input_ids"].clone().detach()
    token_type_ids = inputs["token_type_ids"].clone().detach()
    attention_mask = inputs["attention_mask"].clone().detach()
    
    return input_ids, token_type_ids, attention_mask

def tokenize_one_batch(tokenizer, input_texts: list):
    input_ids = []
    token_type_ids = []
    attention_masks = []
    for text in input_texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True, # Add [CLS]- and [SEP]-Tokens
                            max_length = tokenizer.model_max_length,
                            padding = 'max_length',
                            truncation=True,
                            return_token_type_ids = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        
        # Add the encoded sentence to the list 
        input_ids.append(encoded_dict['input_ids'])
        
        # Add encoded token type index to the list 
        token_type_ids.append(encoded_dict['token_type_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    
    return input_ids, token_type_ids, attention_mask
    