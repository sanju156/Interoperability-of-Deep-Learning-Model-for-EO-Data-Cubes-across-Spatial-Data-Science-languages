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

* Execute the commands in the onnx-container/ directory:

       - docker build -t onnx-repro-base -f Dockerfile.base . (BASE IMAGE)
       - docker build -t onnx-py-julia-train -f Python_Julia_Train_Dockerfile.dockerfile .
       - docker build -t onnx-rtrain -f R_Train_Dockerfile.dockerfile .

* Tag and push the Docker image to the Docker Hub : 

       - docker tag onnx-py-julia-train <dockerhub_name>/onnx-py-julia-train:v1
       - docker push <dockerhub_name>/onnx-py-julia-train:v1

* In HPC, connect to the compute node:

       1. Load the Singularity module: module load Singularity/1.2.5
       2. Build the Singularity container by taking the Docker image from the Docker Hub:  singularity build onnx-py-julia-train.sif docker://<dockerhub_name>/onnx-py-julia-train:v1. This creates the .sif file in the working directory.

* Create an interactive job with GPU support and allocate necessary memory:

Eg: srun --partition=gpuexpress --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=05:00:00 --pty bash

        1. Load the Singularity module
        2. singularity shell --nv --writable-tmpfs onnx-py-julia-train.sif
        3. Launch the Jupyter notebook (jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='')
        4. ssh -i "$HOME/.ssh/id_ecdsa_palma" -o MACs=hmac-sha2-256 -L 8890:r10n06(compute node name):8888 <palma_login_id> which can connect to the jupyter notebook.

* For the Julia GPU setup, execute the commands below inside the container to run the scripts

* Singularity runs containers as a non-root user, and the default Julia depot inside the image is read-only.

* Executing the scripts via Jupyter Notebook in the Sigularity Container can't be done due to a non-root user; instead, use .jl scripts. Whereas in the local machine, you can execute the Jupyter notebook scripts in the Docker container.

* A custom depot in /tmp allows package precompilation and GPU runtime initialisation.

        - export JULIA_DEPOT_PATH="/tmp/julia_cache:/opt/julia_depot"
        - mkdir -p /tmp/julia_cache

* Verify the Julia environment
  
        - julia --project=/app/julia -e 'using Pkg; Pkg.status(); using Flux'
        - julia -e 'using CUDA, cuDNN; CUDA.set_runtime_version!(v"12.1"); cuDNN.version()'


* R (torch) GPU execution: When running R training with GPU support inside a Singularity container on PALMA II, additional setup is required to allow {torch} to download and access CUDA-enabled binaries (“lantern” files).

        - export TORCH_HOME=/home/palma_login_name/.local/torch
        - mkdir -p $TORCH_HOME
        In R
        Sys.setenv(TORCH_HOME = Sys.getenv("TORCH_HOME"))
        Sys.setenv(TORCH_INSTALL_CUDA = "1")
        library(torch)
        install_torch()
        cuda_is_available()
  
## Docker Workflow
<img width="931" height="556" alt="image" src="https://github.com/user-attachments/assets/f6a743f4-66f2-46a3-9569-50cff246ba85" />




  
  



