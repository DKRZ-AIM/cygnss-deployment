#!/bin/bash

git clone --recurse-submodules https://gitlab.dkrz.de/aim/cygnss-deployment

cd cygnss-deployment/docker_cygnss_deployment

docker-compose up --build
