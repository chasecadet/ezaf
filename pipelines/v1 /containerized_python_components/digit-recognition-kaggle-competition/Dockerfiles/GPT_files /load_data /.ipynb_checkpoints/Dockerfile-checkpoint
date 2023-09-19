# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /usr/src/app

# Install necessary libraries
RUN pip install pandas numpy

# Copy the current directory contents into the container
COPY ./load_data.py .

# Run the script when the container launches
CMD ["python", "./load_data.py"]
