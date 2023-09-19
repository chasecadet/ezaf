# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /usr/src/app

# Install necessary libraries
RUN pip install pandas numpy scikit-learn tensorflow<3.0

# Copy the current directory contents into the container
COPY ./prediction.py .

# Run the script when the container launches
CMD ["python", "./prediction.py"]
