# --- Required to install specific versions ---
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org")
}

# --- Package versions for reproducibility ---
pkgs <- list(
  dplyr       = "1.1.3",
  geojsonsf   = "2.0.3",
  ggplot2     = "3.4.4",
  jsonlite    = "1.8.7",
  readr       = "2.1.4",
  Matrix      = "1.5-4.1",
  purrr       = "1.0.2",
  kableExtra  = "1.4.0",
  torch       = "0.15.1"
)

# --- Install pinned versions ---
for (pkg in names(pkgs)) {
  remotes::install_version(pkg, version = pkgs[[pkg]], repos = "https://cloud.r-project.org")
}

# --- Install additional packages (latest version is fine) ---
install.packages(c(
  "progress", "coro", "yardstick", "rsample", "hdf5r",
  "MLmetrics", "vcd", "caret", "tibble", "reticulate", 
  "gplots", "gridExtra", "glue", "geometries","rlang", "magrittr","luz"
), repos = "https://cloud.r-project.org")

if (!requireNamespace("IRkernel", quietly = TRUE)) {
  install.packages("IRkernel", repos = "https://cloud.r-project.org")
}
IRkernel::installspec(user = FALSE)