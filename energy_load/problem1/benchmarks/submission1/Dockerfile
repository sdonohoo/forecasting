#Download base image
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
WORKDIR /research
#Update Ubuntu software repository
RUN apt-get update
#Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip
ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN apt-get install -y python-setuptools
RUN apt-get install -y python-pip python3-pip virtualenv htop vim
#RUN pip3 install --upgrade numpy scipy sklearn tf-nightly-gpu==1.9.0.dev20180419
#Mount requirement.txt into the docker
ADD . /research
#Install additional packages in the txt file
RUN pip3 install -r requirements.txt
RUN cd ..
ENTRYPOINT ["/bin/bash"]
