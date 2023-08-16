# Dockerfile to set up MXNet for CPU
FROM ubuntu:20.04

# Avoid tzdata prompt for timezone
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    python3-pip \
    python3-dev \
    gcc \
    git \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for latest packages
RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.19
# Install MXNet Python package (CPU version)
RUN pip3 install mxnet

# Set the default Python version to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

CMD ["python", "-c", "import mxnet; print(mxnet.__version__)"]
# Clone MXNet
RUN git clone https://github.com/apache/incubator-mxnet.git -b v1.6.x