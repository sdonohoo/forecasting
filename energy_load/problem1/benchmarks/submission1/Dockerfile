# Download base image
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
WORKDIR /tmp

# Update Ubuntu software repository
RUN apt-get update

# Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip
ENV HOME /tmp
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN apt-get install -y python-setuptools
RUN apt-get install -y python-pip python3-pip virtualenv htop vim

# Mount requirements.txt into the docker
ADD ./requirements.txt /tmp

#Install additional packages in the txt file
RUN pip3 install -r requirements.txt
WORKDIR /
RUN rm -rf tmp
ENTRYPOINT ["/bin/bash"]
