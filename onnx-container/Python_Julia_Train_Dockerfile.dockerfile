# Start from the unified base image with R, Python, and Julia support
FROM onnx-repro-base

# Set the working directory
WORKDIR /app

# --- Copy files into container ---
# Python scripts
COPY Python_train/ ./Python_train/

# R script
COPY R_infer/ ./R_infer/

# Julia script
COPY Julia_train/ ./Julia_train/

# Model output directory (copy it if needed)
#COPY model/ ./model/

# Set permissions 
RUN chmod -R a+rw /app

# Default command 
CMD ["bash"]
