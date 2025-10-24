import xarray as xr

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import time
import json

from smallCNN import smallCNN

import mlflow


# Normalize each 2D array to range [0, 1] using its own min/max
def normalize_to_range(tensor):
    """
    Normalize each 2D array to range [0, 1] using its own min/max
    tensor shape: (N, H, W) where N is number of samples
    """
    normalized_list = []
    for i in range(tensor.size(0)):
        array_2d = tensor[i]
        max_val = array_2d.max()
        min_val = array_2d.min()
        range_val = max_val - min_val
        normalized_array = (array_2d - min_val) / range_val
        normalized_list.append(normalized_array)
    
    return torch.stack(normalized_list)

def data_processing(params):
    '''
    Load and process the dataset from NETCDF file.
    Create training and validation dataloaders.'''
    #read the NETCDF file
    print('Reading the NETCDF dataset...')
    #Xarray reads the NETCDF lazily. Only when an operation is needed the values are loaded.
    ds_kappa_psi=xr.open_dataset(params["train_data_path"])
    print(ds_kappa_psi)

    #DEfine the varibales in the model
    y=ds_kappa_psi['kappa'].values
    x=ds_kappa_psi['psirz'].values

    y_tensor=torch.tensor(y)
    x_tensor=torch.tensor(x)
    x_tensor = torch.unsqueeze(x_tensor,1) #add channel dimension (to be consistent with Pytorch conv2d)

    #normalize the data
    print('Normalizing the dataset...')
    #normalize x (flux surface) to [0,1] range for more stable training
    x_tensor_normalized = normalize_to_range(x_tensor)

    #normalize y
    y_mean = y_tensor.mean()
    y_std = y_tensor.std()
    y_tensor_normalized = (y_tensor - y_mean) / y_std

    dataset=TensorDataset(x_tensor_normalized,y_tensor_normalized)


    #split in to train(70%) and validation(30%) datasets
    train_dataset, val_dataset = random_split(dataset,[0.7,0.3])

    #create the dataloaders
    batch_size=params["batch_size"]
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,
                            pin_memory=True,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=batch_size,
                            pin_memory=True)
    return train_dataloader, val_dataloader



print('    ')
print('----------------------------------')
print('Starting the training script')
print('----------------------------------')
print('    ')

#check if GPU is available
print('Checking for GPU availability...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('We are using:',device)

print('Reading the hyperparameters from json file...')
#read hyperparameters from json file
with open('parameters_train.json') as f:
    params = json.load(f)

print('Parameters used for training:')
for key, value in params.items():
    print('  {}: {}'.format(key, value))
print('    ')

#Load and process the data. Create dataloaders
print('Loading and processing the dataset...')
train_dataloader, val_dataloader = data_processing(params)


#calculate the number of 'steps' in the training(validation) loop for future calculations
batch_size=params["batch_size"]
train_steps=len(train_dataloader.dataset)//batch_size
val_steps=len(val_dataloader.dataset)//batch_size


#hyperparameters
learning_rate=params["learning_rate"]
epochs=params["epochs"]

print('Initializing the model, optimizer, and loss function...')
model=smallCNN()
#create a resnet18 model
#model=resnet.ResNet(resnet.ResnetBlock,[2,2,2,2])

model.to(device)
optim=torch.optim.Adam(model.parameters(),lr=learning_rate)

loss_fn=torch.nn.MSELoss()

train_loss_list=[]
val_loss_list=[]
total_time=0

print('Starting the training for {} epochs...'.format(epochs))
print('----------------------------------')
for e in range(epochs):
    t0=time.time()

    #train
    model.train()
    train_loss=0
    for fluxsurf,kappa in train_dataloader:
        fluxsurf = fluxsurf.to(device)
        kappa = kappa.to(device)
        predict = model(fluxsurf)
        predict=torch.squeeze(predict,dim=1)
        loss=loss_fn(predict,kappa)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss+=loss.item()
    train_loss=train_loss/train_steps #average loss in the train loop

    #validation
    val_loss=0
    with torch.no_grad():
        model.eval()
        for fluxsurf,kappa in val_dataloader:
            fluxsurf = fluxsurf.to(device)
            kappa = kappa.to(device)
            predict=model(fluxsurf)
            predict=torch.squeeze(predict,dim=1)
            loss=loss_fn(predict,kappa)
            val_loss+=loss.item()
    val_loss=val_loss=val_loss/val_steps

    epoch_time=time.time()-t0
    total_time+=epoch_time

    print('Current epoch:{}, Epoch time:{:.2f}s,  training loss:{:.4f},  validation loss:{:.4f}'.format(e+1,epoch_time,train_loss,val_loss))
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
print(' ')
print('----------------------------------')
print('Training completed.')
print('Total training time:{:.2f}s'.format(total_time))

#saving the model
print('Saving the trained model to smallCNN_model.pth ...')
torch.save(model.state_dict(),'smallCNN_model.pth')

#LOGGING WITH MLFLOW
print('Logging the results with MLflow...')
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Set the experiment name
mlflow.set_experiment('kappa_prediction_CNN_experiment')

#set run name with date and time
run_name = 'run_{}'.format(time.strftime("%Y%m%d-%H%M%S"))

with mlflow.start_run(run_name=run_name):
    # Log hyperparameters
    mlflow.log_params(params)

    #log training and validation loss lists
    for i in range(len(train_loss_list)):
        mlflow.log_metric("train_loss", train_loss_list[i], step=i)
        mlflow.log_metric("val_loss", val_loss_list[i], step=i)

    # Log final training and validation loss
    mlflow.log_metric("final_train_loss", train_loss_list[-1])
    mlflow.log_metric("final_val_loss", val_loss_list[-1])

    # Log the model
    mlflow.pytorch.log_model(model, "smallCNN_model")