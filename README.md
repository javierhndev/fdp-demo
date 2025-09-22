# A demostration of the FDP capabilities
In this repository, an example of a typical use of the Fusion Data Platform (FDP) is shown. The goal is to create a model that predicts the values of plasma elongation given the flus surface values.

For this, we can get those measurements from the DIII-D database using Toksearch, save them on a netCDF file and then build a neural network in Pytorch that will train on that given dataset.

This task has been performed on Expanse (SDSC). An interactive job is used to run the script that genererates the dataset. Then a Jupyter Notebook is used for data analysys and modeling using Expanse's galyleo.


TO DO list:
- Test a single environment for dataset and modeling.
- The same warning appear multiple times during dataset creation (when concatenating)

## Building the environment
We used two different Conda environments for this demo (we keep them separated to avoid conflicts)

### Dataset environment
To generate a dataset we need first to create an environment with Toksearch following the same instructions as in [their own repo](https://github.com/GA-FDP/toksearch_d3d)

```bash
conda create -n fdp-toksearch -c ga-fdp -c conda-forge toksearch_d3d matplotlib
```
(`matplotlib` has been added for data analysis)


### Modeling environment
A second environment can be created for the analysis of the netCDF files and modeling. We use the environment described in `environment_modeling.yml` to execute the `modeling.ipynb` notebook. To install that environment simply do:

```bash
conda env create -f environment_modeling.yml
``` 



## Create your own dataset with Toksearch

The script `dataset_gen.py` uses Toksearch to pull the DIII-D database and save the elongation (*kappa*) and flux surface (*psirz*) from roughly 1000 shots. Each shot contains different number of time slices which leads to a total number of points of 200k+.

To run the script, within the Conda environment, one can do:
```bash
fdp run python dataset_gen.py
```

The script uses `pipeline` and `MdsSignal` to access the DIII-D dataset. The data is then fetched into an xarray Dataset for each shot. *Kappa* is stored in a 1D array where each value correspond to a time slice. Meanwhile *psi* is a 2D array of the *psi* values in a cross section.
 
The Datasets from each shot are combined into a larger one that it is finally saved on a netCDF file.

## Modeling: A regression model to predict elongation

In the `modeling.ipynb` notebook, a regression model is built to predict plasma elongation (*kappa*) from flux surface using the dataset described above. 

In this example we take a smaller subset (50k points) from the Dataset as enough to train the model and save memory during computation.

A simple convolutional neural network is built with Pytorch for this regressions task. There is a hughe imbalance on the dataset so most of the values are close  to *kappa=1.7*. This causes the model to predict values around 1.7 too.
