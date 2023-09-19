# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /usr/src/app

# Install necessary libraries
RUN pip install wget pandas<2.0.0 numpy scikit-learn

# Copy the current directory contents into the container
COPY ./download_load_preprocess_data.py .

# Run the script when the container launches
CMD ["python", "./download_load_preprocess_data.py"]