# Use the official Python image as the base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies and Jupyter Notebook
RUN pip install -r requirements.txt && \
    pip install jupyter
    
# Install system dependencies for graphical applications
RUN apt-get update && apt-get install -y libgl1-mesa-glx strace

# Copy the Jupyter Notebook file into the container
COPY scannet_aug.ipynb /app/scannet_aug.ipynb

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook when the container is launched
CMD ["strace", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
