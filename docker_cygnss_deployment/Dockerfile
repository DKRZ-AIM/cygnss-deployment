FROM python:3.9
FROM continuumio/miniconda3
WORKDIR app/
COPY requirements.txt .
RUN conda install -c conda-forge cartopy
RUN conda install xarray=0.20.1
RUN pip install -r requirements.txt
