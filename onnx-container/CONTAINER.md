# Docker Execution

This directory [onnx-container/](onnx-container) provides Dockerfiles and scripts for reproducible TempCNN training, ONNX export, and cross-language inference (Python/PyTorch, R/torch, Julia/Flux) using BreizhCrops data. The containers isolate all dependencies to support FAIR workflows and are executed on the PALMA II HPC cluster at the University of Münster.

## Directory Overview

onnx-container/
├── Julia_train/          # Julia (Flux) training, ONNX export & inference scripts
├── Python_train/         # Python (PyTorch) training, ONNX export & inference scripts
├── R_train/              # R (torch) training scripts 
├── R_infer/              # R cross-language ONNX inference/native inference scripts 
├── julia/                # Julia deps 
├── Dockerfile.base       # Base image (Python/R/Julia/system deps)
├── Python_Julia_Train_Dockerfile.dockerfile  # Py+Julia training container
├── R_Train_Dockerfile.dockerfile             # R training container
├── install_r.R         # R deps
└── requirements.txt    # Python deps

