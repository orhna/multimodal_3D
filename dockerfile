# Use the Ubuntu base image
FROM ubuntu:latest

# Install system dependencies for graphical applications and Git
RUN apt-get update && apt-get install -y libgl1-mesa-glx git

# Install Python3 and pip3
RUN apt-get install -y python3 python3-pip

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install Git repository and Python dependencies
RUN pip3 install -r requirements.txt && \
    pip3 install jupyter

# Copy the Jupyter Notebook file into the container
COPY aug_projection.ipynb /app/aug_projection.ipynb

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook when the container is launched
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
