import xarray as xr
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader 

import json
import time

from smallCNN import smallCNN

#import mlflow
from cmflib import cmf

import matplotlib.pyplot as plt

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

def denormalize(kappa, mean, std):
    '''
    Denormalize the kappa values using the mean and std
    '''
    #convert list to numpy
    kappa = np.array(kappa)
    return kappa * std + mean

def data_processing(params):
    '''
    Load and process the dataset from NETCDF file.
    Create test dataloaders (and save normalization parameters).'''
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
    batch_size=params["testing"]["batch_size"]
    test_dataloader=DataLoader(dataset,batch_size=batch_size,
                            pin_memory=True)
    return test_dataloader, y_mean.item(), y_std.item()

print('    ')
print('----------------------------------')
print('Starting the testing script')
print('----------------------------------')
print('    ')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Reading the hyperparameters from json file...')
#read hyperparameters from json file
with open('parameters.json') as f:
    params = json.load(f)

#Load and process the data. Create dataloaders
print('    ')
print('Loading and processing the dataset...')
test_dataloader, kappa_mean, kappa_std = data_processing(params)

#calculate the number of 'steps' in the testing loop for future calculations
batch_size=params["testing"]["batch_size"]
test_steps=len(test_dataloader.dataset)//batch_size

print(' ')
print('Initializing the model and loss function...')
model=smallCNN()
#restore the model weight from file
model.load_state_dict(torch.load(params["model_weights_path"], weights_only=True))

#move the model to the device (GPU or CPU)
model.to(device)

loss_fn=torch.nn.MSELoss()

#testing
test_loss=0
kappa_preds=[]
kappa_trues=[]
with torch.no_grad():
    model.eval()
    for fluxsurf,kappa in test_dataloader:
        fluxsurf = fluxsurf.to(device)
        kappa = kappa.to(device)
        predict=model(fluxsurf)
        predict=torch.squeeze(predict,dim=1)
        loss=loss_fn(predict,kappa)
        test_loss+=loss.item()
        kappa_preds.extend(predict.cpu().tolist())
        kappa_trues.extend(kappa.cpu().tolist())
    test_loss=test_loss/test_steps

print('Test MSE Loss: {:.4f}'.format(test_loss))

print(' ')
print('Plotting predictions vs true values...')
kappa_preds=denormalize(kappa_preds, kappa_mean, kappa_std)
kappa_trues=denormalize(kappa_trues, kappa_mean, kappa_std)

#plotting some predictions vs true values
x_aux=np.linspace(1,2,10)
y_aux=x_aux
plt.figure(figsize=(8,6))
plt.scatter(kappa_trues, kappa_preds)
plt.plot(x_aux,y_aux,color='red',linestyle='--',label='Prediction=True')
plt.xlabel('True elongation')
plt.ylabel('Predicted elongation')
plt.title('Predicted vs True Elongation on Test Set')
plt.xlim(0.8,2.2)
plt.ylim(0.8,2.2)
plt.legend()
plt.savefig('kappa_prediction_test.png')
#plt.show()


metawriter = cmf.Cmf(filepath="mlmd", pipeline_name="FDP-demo")
_ = metawriter.create_context(pipeline_stage="Evaluate")
_ = metawriter.create_execution(execution_type="Evaluate-execution")
_ = metawriter.log_model(path=params["model_weights_path"], event="input",
                         model_framework="Pytorch",model_type="CNN",model_name="Rsmall_CNN")
_ = metawriter.log_dataset(params["test_data_path"], "input")
_ = metawriter.log_execution_metrics("test loss", dict(test_loss=test_loss))

