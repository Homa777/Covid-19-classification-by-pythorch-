#!/usr/bin/env python
# coding: utf-8

# ## Training

# In[1]:


# Define a function to train our model
def train_model(model, criterion, optimizer, scheduler, batch_szie, num_epochs):
    # The number of epochs is a hyperparameter that defines 
    # the number times that the learning algorithm will work through the entire training dataset
    since = time.time() 
    
    best_model_wts = copy.deepcopy(model.state_dict()) # Copy weights and biases of the pre-trained model into our model
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step() ## Adjust the learning rate based on the number of epochs
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
# Set the initial value of variables 
            best_acc = 0.0
            train_acc= list()
            valid_acc= list()
            running_loss = 0.0
            running_corrects = 0
            running_prec= 0.0
            running_rec = 0.0
            running_f1  = 0.0

            # Iterate over data.
            cur_batch_ind= 0
            for inputs, labels in dataloaders[phase]:
                print(cur_batch_ind,"batch inputs shape:", inputs.shape)
                print(cur_batch_ind,"batch label shape:", labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): ## It makes sure to clear the intermediate values
                    ## for evaluation, which are needed to backpropagate during training, 
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Returns the maximum value of all elements in the input tensor
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() # hold the current state and will update the parameters based on the computed gradient

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                cur_acc= torch.sum(preds == labels.data).double()/batch_szie
                cur_batch_ind +=1
                print("\npreds:", preds)
                print("label:", labels.data)
                print("%d-th epoch, %d-th batch (size=%d), %s acc= %.3f \n" %(epoch+1, cur_batch_ind, len(labels), phase, cur_acc ))
                
                if phase=='train':
                    train_acc.append(cur_acc) ## append all the accuracy related to each epoch
                else:
                    valid_acc.append(cur_acc)
            
            ## calculation of accuracy for validation data   
            
            epoch_loss= running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f} \n\n'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch= epoch
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc= %.3f at Epoch: %d' %(best_acc,best_epoch) )

   # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, valid_acc


# In[ ]:




