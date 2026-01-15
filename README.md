# A demostration of the FDP capabilities
In this repository, an example of a typical use of the Fusion Data Platform (FDP) is shown. In this demostration the goal is to create a model that predicts the values of plasma elongation given a cross section of the flux surfaces in DIII-D. All process is tracked using [CMF](https://hewlettpackard.github.io/cmf/).

We are providing a Jupyter Notebook `fdp-demonstration.py` that will guide you on all the process. There are fours steps:
- Data extraction using TokSearch.
- (Optional) Data analysis.
- Model training.
- Model testing.

The `parameters.json` file is used to set the (hyper-)parameters for all the workflow.

This example (without CMF) has been executed on Expanse (SDSC) although any computer would be OK. If having issues with RAM, try to reduce dataset size.

**Prerequisites**:
- Linux (Ubuntu/Debian)
- (Mini)Conda
- Git
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu) with [non-root user](https://docs.docker.com/engine/install/linux-postinstall/) privileges
- [Docker Compose Plugin](https://docs.docker.com/compose/install/linux/)

**TO DO** list:
- CMF integration in Expanse (or other HPC)

## Building the environment
We use a Conda environment for this demo. The requirements are specified in `environment.yml` file. The versions of the tested packages have been added but the only hard requiste is `Python<3.12` due `CMF`. It will install [TokSearch](https://github.com/GA-FDP/toksearch_d3d), [CMF](https://hewlettpackard.github.io/cmf/) and other commons packages in AI/ML (you may need to modify the Pytorch version to your CUDA).

To install that environment simply do:

```bash
conda env create -f environment.yml
``` 

## FDP demostration

The `fdp-demonstration.ipynb` will guide you through all of this but here we are summarizing the sections in the Notebook. Alternatively you can execute use `run.sh` from bash to run the whole demo (after CMF has been initialized).

### Initialize CMF
CMF needs to be initialized to run this demo. It will keep track of the dataset,models and other metadata.

**First step: set the CMF server**

First clone the CMF repository (don't need to be in this folder):
```bash
git clone https://github.com/HewlettPackard/cmf
```
then
```bash
cd cmf
```
then set the environment configuration. Create a .env file in the same directory as `docker-compose-server.yml` with the following environment variables:
```bash
CMF_DATA_DIR=yourhomefolder/fdp-demo/server                    
NGINX_HTTP_PORT=80                  
NGINX_HTTPS_PORT=443
REACT_APP_CMF_API_URL=http://your-server-ip:80
```
Start the containers:
```bash
docker compose -f docker-compose-server.yml up
```
Once the containers are successfully started, the CMF UI will be available at the URL specified in your .env file:
```bash
http://your-server-ip:80
```
**Next step: Initialize CMF**

 Execute the following command (modify the repo to your own):
```bash
cmf init local --path . --git-remote-url https://github.com/javierhndev/fdp-cmf-artifacts.git
```
With current setting, during workflow execution, CMF will create a new branch (`mlmd`) in the git repository where it will store the metadata.

### Create your own dataset with Toksearch

The script `src/dataset_gen.py` uses Toksearch to pull the DIII-D database and save the elongation (*kappa*) and flux surface cross section (*psirz*) from roughly 1000 shots. Each shot contains different number of time slices which leads to a total number of points of 200k+.

To run the script, within the Conda environment, one can do:
```bash
fdp run python dataset_gen.py
```

The script uses `pipeline` and `MdsSignal` to access the DIII-D dataset. The data is then fetched into an `xarray` Dataset for each shot. *Kappa* is stored in a 1D array where each value correspond to a time slice. Meanwhile *psi* is a 2D array of the *psi* values in a cross section.

The following image shows the data from an arbitraty shot and time
![plot](cross_section.png)


 
The Datasets from each shot are combined into a larger one that it is finally saved on a two netCDF files for training and testing.

### Model training

In `src/train.py`, a regression model is used to predict plasma elongation (*kappa*) from flux surface using the dataset described above. 

A simple convolutional neural network (CNN) is built with Pytorch for this regressions task. With `MLflow`, some metrics such as loss are saved to track model performance.

The script will output the model weights in a a figure with the training and validation losses.

### Model testing

In `src/test.py` the model is tested. The saved model is loaded and evaluated (loss) using the *testing* dataset that was generated in the first step. The Notebook includes a figure where the predicted and ground truth are plotted.

### Pushing the metada to the CMF server
After running the model you can save the metadata/artifacts to the CMF server. Execute the following:
``bash
cmf metadata push -p FDP-demo
cmf artifact push -p FDP-demo
```

You can then access the CMF UI to visualize the workflows:
```bash
http://your-server-ip:80
```

If you want to stop the containers with the CMF server:
```bash
docker compose -f docker-compose-server.yml stop
```
