# The following script uses Toksearch to get the psi and kappa values from 1000 shots (at every time slice).
# The values are then stored in a netCDF file.
# Shots are selected randomly from a given range.

from toksearch import MdsSignal, Pipeline
import xarray as xr
import numpy as np

import time
import json

from cmflib import cmf

def create_pipeline(shots):

    pipe = Pipeline(shots)

    kappa_signal = MdsSignal(r"\kappa", "efit01")

    dims = ("r", "z", "times")
    psirz_signal = MdsSignal(r'\psirz', 'efit01', dims=dims, data_order=['times', 'z', 'r'])

    pipe.fetch_dataset("ds", {"kappa": kappa_signal, "psirz": psirz_signal})

    #drop shots with errors (or non exisiting shot)
    @pipe.where
    def no_errors(rec):
        return not rec.errors

    return pipe


def concatenate(pipe):
    '''
    MdsSignal pipelines return a list of datasets (one per shot).
    This functions concatenates the list of datasets in the pipeline into a single xarray Dataset
    along the time dimension.
    '''
    list_of_ds=[pipe[0]['ds']]
    for i in range(1,len(pipe)): #starts from the second index
        if not pipe[i]['ds']['psirz'].isnull().any(): #check for nans and skip ds if any
            #get last value from last dataset
            lastime=pipe[i-1]['ds']['times'][-1]
            #redefine times in the second Dataset
            pipe[i]['ds']['times'] = pipe[i]['ds']['times']+lastime
            #append
            list_of_ds.append(pipe[i]['ds'])
    ds_concat=xr.concat(list_of_ds,dim='times',join='outer')
    return ds_concat


if __name__ == "__main__":
#read hyperparameters from json file
    with open('parameters.json') as f:
        params = json.load(f)
    first_shot=params["dataset_generation"]["first_shot"]
    last_shot=params["dataset_generation"]["last_shot"]
    numshots_train=params["dataset_generation"]["numshots_train"]
    numshots_test=params["dataset_generation"]["numshots_test"]

    print('    ')
    print('Creating the TRAINING dataset with Toksearch...')
    print('    ')
    t0=time.time()

    #TRAINING DATASET
    shot_list=np.random.randint(first_shot,last_shot,numshots_train)
    print('Shot list length for the training dataset:',len(shot_list))
    
    print('Creating the pipeline')
    pipe_train = create_pipeline(shot_list)

    print('Computing the pipeline')
    results = pipe_train.compute_multiprocessing()
    #results = pipe.compute_serial()

    print('Concatenating the dataset')
    ds_concat = concatenate(results)

    print('We got {} shots'.format(len(results)))
    #print(results[0]['ds'])
    print(' ')
    print('The final training Dataset with all shots is:')
    print(ds_concat)
   
    print('Writing to netCDF')
    #write to netCDF
    ds_concat.to_netcdf(params["train_data_path"])

    #ds_read=xr.open_dataset('kappa_psi_dataset.nc')
    #print (' Read DS ')
    #print(ds_read)
    print('Total time for creating the training dataset: {:.2f} seconds'.format(time.time()-t0))
    print('    ')

    metawriter = cmf.Cmf(filepath="mlmd", pipeline_name="FDP-demo")
    _ = metawriter.create_context(pipeline_stage="Prepare", custom_properties=params["dataset_generation"])
    _ = metawriter.create_execution(execution_type="Prepare")

    _ = metawriter.log_dataset(params["train_data_path"], "output")

    #################
    print('    ')
    print('Creating the TESTING dataset with Toksearch...')
    print('    ')
    t1=time.time()

    #TESTING DATASET
    shot_list_test=np.random.randint(first_shot,last_shot,numshots_test)
    print('Shot list length for the testing dataset:',len(shot_list_test))

    pipe_test = create_pipeline(shot_list_test)

    print('Computing the pipeline')
    results_test = pipe_test.compute_multiprocessing()

    print('Concatenating the dataset')
    ds_concat_test = concatenate(results_test)

    print('We got {} shots'.format(len(results_test)))

    print('The final testing Dataset with all shots is:')
    print(ds_concat_test)

    print('Writing to netCDF')
    #write to netCDF
    ds_concat_test.to_netcdf(params["test_data_path"])

    #ds_read=xr.open_dataset('kappa_psi_dataset.nc')
    #print (' Read DS ')
    #print(ds_read)
    print('Total time for creating the testing dataset: {:.2f} seconds'.format(time.time()-t1))

    _ = metawriter.log_dataset(params["test_data_path"], "output")
    metawriter.finalize()