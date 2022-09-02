# Usage

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
