# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /usr/src/app

# Install wget library
RUN pip install wget

# Copy the current directory contents into the container
COPY ./download_data.py .

# Run the script when the container launches
CMD ["python", "./download_data.py"]