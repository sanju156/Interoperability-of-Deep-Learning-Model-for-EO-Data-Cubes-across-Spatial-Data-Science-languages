# Interoperability-of-Deep-Learning-Model-for-Earth Observation-Data-Cubes-across-Spatial-Data-Science-languages
This repository contains the code, trained models, and evaluation results for the Master's thesis.

# Overview
This work focuses on reproducibility and cross-language interoperability of deep learning models for Earth Observation (EO) data cubes, with a particular emphasis on Temporal Convolutional Neural Networks (TempCNN) for satellite time-series classification.

Three native implementations of the TempCNN architecture are provided:

1. Python – **native-models/pytorch**

2. R  – **native-models/rtorch**

3. Julia – **native-models/flux**

The repository supports a unified workflow:
Train in one language → export to ONNX → import and infer in another, enabling systematic comparison of:

* Cross-language model interoperability (Python, R, Julia)

* Structural integrity and parameter consistency after ONNX export

Prediction equivalence and numerical stability across runtimes

Tooling effort, code portability, and reproducibility constraints
