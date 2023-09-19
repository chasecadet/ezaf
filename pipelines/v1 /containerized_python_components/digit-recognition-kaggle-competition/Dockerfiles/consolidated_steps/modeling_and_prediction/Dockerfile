# Use an official Python runtime as a base image
FROM tensorflow/tensorflow:latest 

# Set the working directory
WORKDIR /usr/src/app

# Install necessary libraries
RUN pip install pandas"<2.0.0" numpy scikit-learn tensorflow"<3.0"

# Copy the current directory contents into the container
COPY ./modeling_and_prediction.py .

# Run the script when the container launches
CMD ["python", "./modeling_and_prediction.py"]
