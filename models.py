"""
@author: Leon Scharwächter
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForSequenceClassification

#%%

class CustomTextClassifier(nn.Module):
    def __init__(self,
                 model_name: str,
                 num_labels: int):
        super(CustomTextClassifier, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.pretrained = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                             num_labels=num_labels,
                                                                             output_hidden_states=True)

    def forward(self, inputs: Tensor,
                token_type_ids: Tensor = [],
                attention_mask: Tensor = [],
                use_embeds: bool = False,
                use_softmax: bool = False):

        if use_embeds == False:
            outputs = self.pretrained(input_ids=inputs,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
        else:
            outputs = self.pretrained(inputs_embeds=inputs,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)
            
        if use_softmax == True:
            return torch.softmax(outputs.logits,dim=1) # needed for Integrated Gradients
        else:
            return outputs # (.logits) needed for the training loss function


'''
# How to initialize:
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 20
model = CustomTextClassifier(model_name=MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)

# Example input:
text = "Wow this method is fantastic and actually works!"

# Tokenize input text and convert to tensor:
inputs = tokenizer(text, return_tensors="pt")
input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

# Forward pass to obtain BERT embeddings:
outputs = model(inputs=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)

logits = outputs[0]
hidden_states = outputs[1]
embeddings = hidden_states[0] 

'''