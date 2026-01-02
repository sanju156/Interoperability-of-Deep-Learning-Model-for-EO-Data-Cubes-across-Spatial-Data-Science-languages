# Interoperability-of-Deep-Learning-Model-for-Earth Observation-Data-Cubes-across-Spatial-Data-Science-languages
This repository contains the code, trained models, and evaluation results for the Master's thesis.

## Overview
This work focuses on the reproducibility and cross-language interoperability of deep learning models for Earth Observation (EO) data cubes, with a particular emphasis on Temporal Convolutional Neural Networks (TempCNNs) for satellite time-series classification.

Three native implementations of the TempCNN architecture are provided:

1. Python – **native-models/pytorch**

2. R  – **native-models/rtorch**

3. Julia – **native-models/flux**

The repository follows a unified experimental workflow for crop-type classification: Train in one language → export to ONNX → import and infer in another, enabling a systematic comparison of:

* Cross-language model interoperability (Python, R, Julia).

* Structural integrity and parameter consistency after ONNX export.

* Prediction equivalence and numerical stability across runtimes.

* Tooling effort and reproducibility constraints.

The PyTorch model implementation used in this study is directly adopted from Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., and Körner, M. (2020) for the BreizhCrops crop-type mapping dataset. The same model architecture was reimplemented in R using Torch and in Julia using Flux to enable cross-language comparison. The dataset and original model code (PyTorch) for crop classification can be found here:  https://github.com/dl4sits/BreizhCrops

## Repository Structure

| Directory         | Description |
|-------------------|-------------|
| `onnx-container/` | Dockerfiles for reproducing the workflow (training, ONNX export, evaluation) |
| `native-models/`  | Trained native models (.pt, .rds, .bson) with STAC-ML metadata |
| `onnx-models/`    | Exported ONNX models used for cross-language inference |
| `data-preprocess/`| Dataset preprocess according to language usability |
| `result-plots/`   | Evaluation figures and comparison plots |
| `references/`     | Link to external dataset and publication |


## Data

The training and validation data used in this work are **not included** here.
Data access procedures and pre-processing according to language usability are present in:
* Data access - [references/Dataset.md]
* Preprocess - [data-preprocess/](data-preprocess)

The original dataset and model code (PyTorch) for crop classification can be found here:  https://github.com/dl4sits/BreizhCrops

## How to reproduce 

1. Clone the repository
2. See the **onnx-container/** for running the Docker container, including the necessary scripts, and mounting the data by accessing from BreizhCrops.
3. STAC-ML metadata are documented for each native model.

## License
MIT License.
See [LICENSE.](LICENSE)
