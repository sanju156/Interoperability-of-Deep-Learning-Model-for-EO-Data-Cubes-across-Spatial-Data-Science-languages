# Use R + tidyverse base
FROM rocker/tidyverse:4.3.2

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    wget curl gnupg unzip git build-essential libssl-dev \
    libcurl4-openssl-dev libxml2-dev libgl1 \
    python3 python3-pip \
    libhdf5-dev libhdf5-serial-dev hdf5-tools \
    && rm -rf /var/lib/apt/lists/*

# --- Install CUDA 11.8 Runtime
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
 && dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
 && cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
 && apt-get update \
 && apt-get install -y cuda-libraries-11-8 \
 && rm cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb


# --- Set CUDA-related environment variables ---
ENV CUDA="11.8"
ENV TORCH_HOME=/root/.local/torch
ENV TORCH_INSTALL=1
ENV TORCH_INSTALL_CUDA=1

# --- GPU-enabled torch ---
RUN Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org'); \
                remotes::install_version('torch', '0.15.1', repos='https://cloud.r-project.org')"


# --- Set TORCH_HOME and create it ---
ENV TORCH_HOME=/root/.local/torch
RUN mkdir -p $TORCH_HOME && chmod -R 777 $TORCH_HOME

# --- Lantern files need to be installed to run the torch ---
RUN R -e "Sys.setenv(TORCH_INSTALL_CUDA='1'); \
  Sys.setenv(TORCH_HOME=Sys.getenv('TORCH_HOME')); \
  library(torch); install_torch()"

# --- R packages ---
RUN R -e "install.packages(c( \
  'progress', 'coro', 'yardstick', 'rsample', \
  'MLmetrics', 'vcd', 'caret', 'tibble', 'reticulate', \
  'gplots', 'gridExtra', 'glue', 'geometries', 'rlang', 'magrittr', \
  'luz', 'geojsonsf', 'jsonlite', 'readr', 'Matrix', 'purrr', 'kableExtra', \
  'dplyr', 'ggplot2', 'psych' \
), dependencies = TRUE)"


# Install hdf5r into system-wide library
RUN Rscript -e "install.packages('hdf5r', lib = .Library)"

RUN pip install --no-cache-dir jupyterlab notebook ipywidgets jupyterlab_widgets \
 && python3 -m ipykernel install --sys-prefix
RUN ["R","-q","-e","if (!requireNamespace('IRkernel', quietly=TRUE)) install.packages('IRkernel', repos='https://cloud.r-project.org'); IRkernel::installspec(user=FALSE)"]


# --- Set the working directory ---
WORKDIR /app
COPY R_train/ ./R_train/
RUN chmod -R a+rw /app
CMD ["bash"]