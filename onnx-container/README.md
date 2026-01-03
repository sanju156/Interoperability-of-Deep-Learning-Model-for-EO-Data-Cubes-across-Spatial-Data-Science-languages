# Docker Execution

This directory [onnx-container/](onnx-container) provides Dockerfiles and scripts for reproducible TempCNN training, ONNX export, and cross-language inference (Python/PyTorch, R/torch, Julia/Flux) using BreizhCrops data. The containers isolate all dependencies to support FAIR workflows and are executed on the PALMA II HPC cluster at the University of Münster.

## Directory Overview

onnx-container/

| File/Directory | Description |
|----------------|-------------|
| `Julia_train/` | Julia (Flux) training, ONNX export & inference scripts |
| `Python_train/` | Python (PyTorch) training, ONNX export & inference scripts |
| `R_train/` | R (torch) training scripts |
| `R_infer/` | R cross-language ONNX inference/native inference scripts |
| `julia/` | Julia deps (Project.toml/Manifest.toml) |
| `Dockerfile.base` | Base image (Python/R/Julia/system deps) |
| `Python_Julia_Train_Dockerfile.dockerfile` | Py+Julia training dockerfile |
| `R_Train_Dockerfile.dockerfile` | R training dockerfile |
| `install_r.R` | R deps |
| `requirements.txt` | Python deps |

## Build and Run the Docker container 

Execute the commands in the onnx-container/ directory:

* docker build -t onnx-repro-base -f Dockerfile.base . (BASE IMAGE)
  
* docker build -t onnx-py-julia-train -f Python_Julia_Train_Dockerfile.dockerfile .
  
* docker build -t onnx-rtrain -f R_Train_Dockerfile.dockerfile .

Tag and push the Docker image to the Docker Hub : 

* docker tag onnx-py-julia-train <dockerhub_name>/onnx-py-julia-train:v1
  
* docker push <dockerhub_name>/onnx-py-julia-train:v1

In HPC, connect to the compute node:

* Load the Singularity module: module load Singularity/1.2.5

* Build the Singularity container by taking the Docker image from the Docker Hub:  singularity build onnx-py-julia-train.sif docker://<dockerhub_name>/onnx-py-julia-train:v1


