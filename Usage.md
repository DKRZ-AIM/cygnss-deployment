# Usage

## In Script

```bash
cd ~cygnss-deployment

# download CyGNSS data
python API.py

# download ERA5 data and annotate CyGNSS data with wind speed labels
# preprocss (filter) to create hdf5
python Preprocessing.py

# Inference
PYTHONPATH="./externals/gfz_cygnss/":${PYTHONPATH}
export PYTHONPATH

python ./externals/gfz_cygnss/gfz_202003/training/cygnssnet.py --load-model-path /work/ka1176/shared_data/2022-cygnss-deployment/cygnss_trained_model/ygambdos_yykDM/trained_model/checkpoint/cygnssnet-epoch\=0.ckpt --data ./dev_data --save-y-true --prediction-output-path ./prediction/current_predictions.h5
```

## In Jupyter notebook

### Kernel

Create `conda` environment using

```bash
conda env create --file docker/kernel-env-cuda11.yaml

conda activate cygnss-d

# some packages were not installed correctly
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch-lightning -c conda-forge
pip install global-land-mask
```
Create Jupyterhub kernel from this environment following https://docs.dkrz.de/doc/software%26services/jupyterhub/kernels.html

### Setup for preprocessing

#### Earthdata

- Retrieve user ID and create `.netrc` as described in ...
- change the persmission of the file: chmod og-rwx ~/.netrc

#### ERA5

Retrieve user ID and API key and create `cdsapi` as described in ...
