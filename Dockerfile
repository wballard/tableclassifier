#base image provides CUDA support on Ubuntu 16.04
FROM nvidia/cuda:8.0-cudnn6-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

#package updates to support conda
RUN apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz

#add on conda python and make sure it is in the path
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet --output-document=anaconda.sh https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
    /bin/bash /anaconda.sh -f -b -p $CONDA_DIR && \
    rm anaconda.sh

#needed packages, using conda for the available match packages
COPY ./*.txt /
RUN conda install --file conda-requirements.txt
RUN pip install --requirement requirements.txt

#volume to have a working data area for training and serving
VOLUME ["/var/data"]

#all the code
COPY ./tableclassifier /tableclassifier



#entrypoint used to train and serve
WORKDIR /
EXPOSE 8888
ENTRYPOINT ["python", "-m", "tableclassifier"]