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

# Install R

ENV R_BASE_VERSION 3.5.1

RUN sh -c 'echo "deb http://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/" >> /etc/apt/sources.list' \
    && gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9 \
    && gpg -a --export E084DAB9 | apt-key add -

RUN apt-get update && apt-get install -y --no-install-recommends r-base=${R_BASE_VERSION}-* \
    && echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >> /etc/R/Rprofile.site

# Install Anaconda
    
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.5.4-Linux-x86_64.sh -b -p /opt/conda

#RUN apt-get install -y python3-pip

WORKDIR /tmp

# Mount python dependency file into the docker container and install
ADD ./python_dependencies.txt /tmp
RUN conda install --file python_dependencies.txt

# Mount R dependency file into the docker container and install

# install dependencies of bayesm
RUN apt-get update && apt-get install -y \
    gfortran \
    libblas-dev \
    liblapack-dev

ADD ./install_R_dependencies.R /tmp
RUN Rscript install_R_dependencies.R

WORKDIR /
RUN rm -rf tmp
ENTRYPOINT ["/bin/bash"]
