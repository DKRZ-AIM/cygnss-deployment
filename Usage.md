# Usage

## In Jupyter notebook

### Kernel

Create `conda` environment using

```bash
conda env create --file docker/kernel-env-cuda11.yaml

conda activate cygnss-d
pip install global-land-mask # for some reason it was not installed otherwise
```
Create Jupyterhub kernel from this environment following https://docs.dkrz.de/doc/software%26services/jupyterhub/kernels.html

### Setup for preprocessing

#### Earthdata

Retrieve user ID and create `.netrc` as described in ...

#### ERA5

Retrieve user ID and API key and create `cdsapi` as described in ...
