FROM python:3.9
FROM continuumio/miniconda3
WORKDIR /app
COPY requirements.txt .
RUN conda install -c conda-forge cartopy
RUN pip install -r requirements.txt

