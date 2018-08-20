FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    zlib1g-dev \
    libssl-dev \
    libssh2-1-dev \
    libcurl4-openssl-dev \
    bzip2 \
    build-essential \
    ca-certificates \
    curl \
    lsb-release \
    apt-utils

# Install Azure CLI

RUN AZ_REPO=$(lsb_release -cs) && \
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | \
        tee /etc/apt/sources.list.d/azure-cli.list && \
    curl -L https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    apt-get install apt-transport-https && \
    apt-get update && apt-get install azure-cli

# Install Anaconda
    
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p /opt/conda

WORKDIR /tmp

# Mount conda dependency file into the docker container and install
ADD ./conda_dependencies.yml /tmp
RUN conda env create --file conda_dependencies.yml

WORKDIR /
ENTRYPOINT ["/bin/bash"]
