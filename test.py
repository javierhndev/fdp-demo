import xarray as xr

import torch
from torch.utils.data import TensorDataset, DataLoader 

import json
import time

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
    Create test dataloaders.'''
    #read the NETCDF file
    print('Reading the NETCDF dataset...')
    #Xarray reads the NETCDF lazily. Only when an operation is needed the values are loaded.
    ds_kappa_psi=xr.open_dataset(params["test_data_path"])
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

    #create the dataloaders
    batch_size=params["batch_size"]
    test_dataloader=DataLoader(dataset,batch_size=batch_size,
                            pin_memory=True)
    return test_dataloader

print('    ')
print('----------------------------------')
print('Starting the testing script')
print('----------------------------------')
print('    ')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Reading the hyperparameters from json file...')
#read hyperparameters from json file
with open('parameters_train.json') as f:
    params = json.load(f)

#Load and process the data. Create dataloaders
print('Loading and processing the dataset...')
test_dataloader = data_processing(params)


#calculate the number of 'steps' in the testing loop for future calculations
batch_size=params["batch_size"]
test_steps=len(test_dataloader.dataset)//batch_size

print('Initializing the model, optimizer, and loss function...')
model=smallCNN()
#restore the model weight from file
model.load_state_dict(torch.load(params["model_weights_path"], weights_only=True))

#move the model to the device (GPU or CPU)
model.to(device)

loss_fn=torch.nn.MSELoss()

#testing
test_loss=0
with torch.no_grad():
    model.eval()
    for fluxsurf,kappa in test_dataloader:
        fluxsurf = fluxsurf.to(device)
        kappa = kappa.to(device)
        predict=model(fluxsurf)
        predict=torch.squeeze(predict,dim=1)
        loss=loss_fn(predict,kappa)
        test_loss+=loss.item()
    test_loss=test_loss/test_steps

print('Test MSE Loss: {:.4f}'.format(test_loss))

if params["tracking"]==True:
    #LOGGING WITH MLFLOW
    print('Logging the results with MLflow...')
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Set the experiment name
    mlflow.set_experiment('kappa_prediction_CNN_experiment')

    # Set run name with date and time
    run_name = 'run_test_{}'.format(time.strftime("%Y%m%d-%H%M%S"))

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(params)

        # Log final testing loss
        mlflow.log_metric("test_loss", test_loss)