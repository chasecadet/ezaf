FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/scripts

# Install all required tools and libraries in one step to reduce layers and optimize build time
RUN apt-get update && apt-get install -y \
    git vim cmake wget sed python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev \
    python3-pip build-essential libopenblas-dev liblapack-dev libopencv-dev software-properties-common gpg \
    libtinfo5 && \
    rm -rf /var/lib/apt/lists/* # Cleanup to reduce image size

# Python packages installation
RUN pip3 install --upgrade pip && \
    pip3 install --user numpy decorator tornado psutil xgboost mxnet

# Clone repositories
RUN git clone --recursive https://github.com/dmlc/tvm

# Download, extract, and setup LLVM
RUN wget http://releases.llvm.org/4.0.0/clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    tar -xf clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    mv clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04 llvm && \
    rm clang+llvm-4.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz

# Set the default Python version to Python 3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install latest CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor > /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/* # Cleanup again

# Compile and install TVM
RUN cd tvm && \
    mkdir build && \
    cp cmake/config.cmake build && \
    cd build && \
    sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ..\/..\/llvm\/bin\/llvm-config)/g' config.cmake && \
    cmake -Wno-dev .. && \
    make -j $(nproc) && \
    cd ../python; python3 setup.py install --user && \
    cd ../topi/python; python3 setup.py install --user && \
    cd ../../nnvm/python; python3 setup.py install --user

# Copy custom code to container
COPY start-job.py .
COPY auto-tuning.py .

# Change working path
WORKDIR /home/log

CMD ["python", "-c", "import mxnet; print(mxnet.__version__)"]
