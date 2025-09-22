# The following script uses Toksearch to get the psi and kappa values from 1000 shots (at every time slice).
# The values are then stored in a netCDF file.
# Shots are selected randomly from a given range.

from toksearch import MdsSignal, Pipeline
import xarray as xr
import numpy as np

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
    ds_concat=pipe[0]['ds']
    for i in range(1,len(pipe)): #starts from the second index
        if not pipe[i]['ds']['psirz'].isnull().any(): #check for nans and skip ds if any
            #get last value from first dataset
            lastime=ds_concat['times'][-1]
            #redefine times in the second Dataset
            pipe[i]['ds']['times'] = pipe[i]['ds']['times']+lastime

            #concat
            ds_concat=xr.concat([ds_concat,pipe[i]['ds']],dim='times')
    return ds_concat


if __name__ == "__main__":

    first_shot=160000#165920
    last_shot=195000#165927
    numshots=1700 #many shot numbers don't exist so need a buffer to get 1000 valid shots.

    #shot_list=np.arange(first_shot,last_shot+1)
    shot_list=np.random.randint(first_shot,last_shot,numshots)
    print('Shot list length:',len(shot_list))
    
    print('Creating the pipeline')
    pipe = create_pipeline(shot_list)
    #pipe = create_pipeline([165920, 165921, 165922])

    print('Computing the pipeline')
    results = pipe.compute_multiprocessing()
    #results = pipe.compute_serial()

    print('Concatenating the dataset')
    ds_concat = concatenate(results)

    print('We got {} shots'.format(len(results)))
    #print(results[0]['ds'])
    print(' ')
    print('The final Dataset with all shots is:')
    print(ds_concat)
   
    print('Writing to netCDF')
    #write to netCDF
    ds_concat.to_netcdf('kappa_psi_dataset.nc')

    #ds_read=xr.open_dataset('kappa_psi_dataset.nc')
    #print (' Read DS ')
    #print(ds_read)
    
