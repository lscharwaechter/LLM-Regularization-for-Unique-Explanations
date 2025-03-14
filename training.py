"""
@author: Leon Scharwächter
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from transformers import TrainingArguments

# import utility scripts
import dataset
import models
import utils

torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:",device,"| index:",device.index)

#%%

# Define pre-trained BERT model and tokenizer
MODEL_NAME = "bert-base-uncased" 

# Define training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-5
PRED_STRATEGY = "multiclass"
EPOCHS = 4

# Define hyperparameters for integrated gradients
N_STEPS = 50

# Define triplet loss hyperparameters
MARGIN = 2

# Define regularization hyperparameters
ALPHA = 1
BETA = 1
GAMMA = 1
DELTA = 1
LAMBDA = 1

#%%

training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=False,
    output_dir=None,
    #fp16=False,
    #weight_decay=0.01,
    #adam_beta1=0.9,
    #adam_beta2=0.999,
    #adam_epsilon=1e-08,
    #learning_rate=LEARNING_RATE,
    #**default_args,
)


#%%

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the dataset
dataloader, labels, num_labels = dataset.getDataloader(MODEL_NAME, BATCH_SIZE)
dataloader_validation, _, _ = dataset.getDataloader(MODEL_NAME, BATCH_SIZE, subset="test")
print("dataset loaded.")

# Load the base model
model = models.CustomTextClassifier(MODEL_NAME, num_labels)
model.to(device)
print("model loaded.")

# Get the function to project the input_ids into the embedding space
if tokenizer.name_or_path == "bert-base-uncased":
    embedding_space = model.pretrained.bert.embeddings.word_embeddings
elif tokenizer.name_or_path == "roberta-base":
    embedding_space = model.pretrained.roberta.embeddings.word_embeddings

#%%

# Initialize optimization strategy
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Initialize classification loss function
if PRED_STRATEGY == "multiclass":
    '''
    In a multi-class classification scenario an input instance belongs to 
    exactly one class. That's why Cross Entropy Loss is used to compare
    the predicted class with the true class label.
    '''
    loss_function = nn.CrossEntropyLoss(reduction="mean")
elif PRED_STRATEGY == "multilabel":
    '''
    In a multi-label classification scenario multiple labels may be 
    assigned independently from each other to one input.
    In this case, the logits are used and a combined sigmoid activation
    and binary cross-entropy loss is applied into a single step.
    '''
    loss_function = nn.BCEWithLogitsLoss(pos_weight=None)

# Initialize triplet loss function (Margin for cosine distance = 2, for pearson correlation = 1)
triplet_loss_function = nn.TripletMarginWithDistanceLoss(distance_function=utils.cosine_distance, margin=MARGIN, reduction='mean')

# Initialize class regularization loss function
js_divergence_loss = utils.JSDivergence(reduction="batchmean")
U = torch.full((BATCH_SIZE, num_labels), 1.0/(num_labels-1), device=device)

#%%

# Perform training
train_errs = []
val_errs = []

classification_errs = []
reg_triplet_cls_errs = []
reg_triplet_att_errs = []
reg_posclass_errs = []
reg_negclass_errs = []

start_time = time.asctime(time.localtime())
for epoch in range(EPOCHS):
    epoch_loss = 0
    epoch_classification_loss = 0
    epoch_triplet_loss_cls = 0
    epoch_triplet_loss_att = 0
    epoch_posclass_loss = 0
    epoch_negclass_loss = 0
    model.train()
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")): 
        input_ids, token_type_ids, attention_mask, labels = batch
        
        input_ids = input_ids.to(device) # Shape: [bz, maxlen]
        token_type_ids = token_type_ids.to(device) # Shape: [bz, maxlen]
        attention_mask = attention_mask.to(device) # Shape: [bz, maxlen]
        labels = labels.to(device) # Shape: [bz]
        
        # Forward pass for the classification loss
        outputs = model(inputs = input_ids,
                        token_type_ids = token_type_ids,
                        attention_mask = attention_mask)
        probabilities = torch.softmax(outputs.logits,dim=1) 
        predictions = probabilities.argmax(-1)
        cls_embeddings = outputs.hidden_states[-1][:,0,:]
               
        # Calculate classification loss
        classification_loss = loss_function(outputs.logits, labels)
        
        # Extract embeddings of current input ids
        input_embeds = embedding_space(input_ids)
        
        # Generate perturbed counterexamples
        input_embeds_perturbed_pos, input_embeds_perturbed_neg = utils.perturb_embeddings(input_embeds, tokenizer, scale=6, scale_neg=7)
        input_embeds_perturbed_pos = utils.restore_system_tokens(input_embeds_perturbed_pos, input_ids, model, tokenizer)
        input_embeds_perturbed_neg = utils.restore_system_tokens(input_embeds_perturbed_neg, input_ids, model, tokenizer)
        
        # Get class predictions and cls embeddings of positive example
        outputs_pos_emb = model(inputs = input_embeds_perturbed_pos,
                                      token_type_ids = token_type_ids,
                                      attention_mask = attention_mask,
                                      use_embeds = True,
                                      use_softmax = False)
        output_pos_emb_logits = outputs_pos_emb.logits
        probabilities_pos_emb = torch.softmax(output_pos_emb_logits,dim=1)
        predictions_pos_emb = probabilities_pos_emb.argmax(-1)
        cls_embeddings_pos = outputs_pos_emb.hidden_states[-1][:,0,:]
        
        # Get class predictions and cls embeddings of negative example
        outputs_neg_emb = model(inputs = input_embeds_perturbed_neg,
                                      token_type_ids = token_type_ids,
                                      attention_mask = attention_mask,
                                      use_embeds = True,
                                      use_softmax = False)
        output_neg_emb_logits = outputs_neg_emb.logits
        probabilities_neg_emb = torch.softmax(output_neg_emb_logits,dim=1)
        predictions_neg_emb = probabilities_neg_emb.argmax(-1)
        cls_embeddings_neg = outputs_neg_emb.hidden_states[-1][:,0,:]
        
        # Initialize integrated gradients w./ current model
        integratedGradientsBatch = utils.IntegratedGradientsBatch(model, tokenizer, device)  
        
        # Get attribution scores
        attributions = integratedGradientsBatch.forward_embeds(input_embeds,
                                                               token_type_ids, 
                                                               attention_mask, labels, n_steps=N_STEPS) #predictions
        attributions_pos = integratedGradientsBatch.forward_embeds(input_embeds_perturbed_pos,
                                                                   token_type_ids,
                                                                   attention_mask, labels, n_steps=N_STEPS) #predictions_pos_emb
        attributions_neg = integratedGradientsBatch.forward_embeds(input_embeds_perturbed_neg, 
                                                                   token_type_ids,
                                                                   attention_mask, predictions_neg_emb, n_steps=N_STEPS)
        
        # Calculate triplet loss
        triplet_loss_cls = triplet_loss_function(cls_embeddings, cls_embeddings_pos, cls_embeddings_neg)
        triplet_loss_att = triplet_loss_function(attributions, attributions_pos, attributions_neg)
        
        # Class regularization for the positive counterexample
        pos_class_loss = js_divergence_loss(probabilities_pos_emb, probabilities)
        
        # Class regularization for the negative counterexample
        U_custom = U.scatter_(1, labels.unsqueeze(1), 0)
        neg_class_loss = js_divergence_loss(probabilities_neg_emb, U_custom)
         
        # Tune total loss
        total_loss = ALPHA*classification_loss + BETA*triplet_loss_cls + GAMMA*triplet_loss_att + DELTA*neg_class_loss + LAMBDA*pos_class_loss
        total_loss = total_loss / training_args.gradient_accumulation_steps
        
        # Accumulate gradients
        total_loss.backward()
        if step % training_args.gradient_accumulation_steps == 0:
            # Perform weight update
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()
        
        epoch_loss += total_loss.item()
        epoch_classification_loss += classification_loss.item()
        epoch_triplet_loss_cls += triplet_loss_cls.item()
        epoch_triplet_loss_att += triplet_loss_att.item()
        
        epoch_posclass_loss += pos_class_loss.item()
        epoch_negclass_loss += neg_class_loss.item()
        
    avg_loss = epoch_loss / len(dataloader) 
    avg_classification_loss = epoch_classification_loss / len(dataloader)
    avg_triplet_cls_loss = epoch_triplet_loss_cls / len(dataloader)
    avg_triplet_att_loss = epoch_triplet_loss_att / len(dataloader)
    avg_posclass_loss = epoch_posclass_loss / len(dataloader)
    avg_negclass_loss = epoch_negclass_loss / len(dataloader)
    
    train_errs.append(avg_loss)
    classification_errs.append(avg_classification_loss)
    reg_triplet_cls_errs.append(avg_triplet_cls_loss)
    reg_triplet_att_errs.append(avg_triplet_att_loss)
    reg_posclass_errs.append(avg_posclass_loss)
    reg_negclass_errs.append(avg_negclass_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # Calculate test error
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_val in dataloader_validation:
            input_ids, token_type_ids, attention_mask, labels = batch_val
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device) 
            attention_mask = attention_mask.to(device) 
            labels = labels.to(device)
            
            outputs_val = model(inputs = input_ids,
                                token_type_ids = token_type_ids,
                                attention_mask = attention_mask)
            
            val_loss = loss_function(outputs_val.logits, labels)
            epoch_val_loss += val_loss.item()
    
        avg_val_loss = epoch_val_loss / len(dataloader_validation)
        val_errs.append(avg_val_loss)
    
    ## Save the current model state (after each epoch)
    torch.save(model.state_dict(), 'regularized_model_epoch'+str(epoch)+'.pt')

# Save the training time to a file
end_time = time.asctime(time.localtime())
with open('training_time.txt', 'w') as f:
    f.write(f"Training started at: {start_time}\n")
    f.write(f"Training ended at: {end_time}\n")

# Save the trained model
np.save('train_errs_reg.npy',train_errs)
np.save('val_errs_reg.npy',val_errs)
np.save('classification_errs.npy',classification_errs)
np.save('triplet_errs_cls_reg.npy',reg_triplet_cls_errs)
np.save('triplet_errs_att_reg.npy',reg_triplet_att_errs)
np.save('posclass_errs_reg.npy',reg_posclass_errs)
np.save('negclass_errs_reg.npy',reg_negclass_errs)

print('Done.')
