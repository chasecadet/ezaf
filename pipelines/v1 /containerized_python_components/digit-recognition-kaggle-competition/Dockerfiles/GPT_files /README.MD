# Dockerfiles 
I assumed that you will separate the functions into individual files named download_data.py, load_data.py, preprocess_data.py, modeling.py, and prediction.py, respectively. Adjust the COPY command if the filenames are different.
Each Dockerfile will be in it's step directory. 
Modify the RUN pip install ... lines to install any other necessary libraries for each step.
Adjust the Python version in the FROM python:3.8-slim line if you are using a different version.
Save each of these Dockerfiles and build them using the docker build -t tag_name . command. After building, you can run them using the docker run tag_name command.