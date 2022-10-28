FROM python:3.9
FROM continuumio/miniconda3
WORKDIR /app
COPY requirements.txt .
RUN conda install -c conda-forge cartopy
RUN pip install -r requirements.txt
RUN prefect deployment build cygnss-deployment/prefect-deploy.py:main -n cygnss -q test
RUN prefect deployment apply cygnss-deployment.yaml
