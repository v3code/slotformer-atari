FROM condaforge/miniforge3
WORKDIR /code/
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install zip build-essential ffmpeg libsm6 libxext6 wget
COPY . .
RUN conda env create -f ./environment.yml
RUN conda init

ENTRYPOINT ["tail", "-f", "/dev/null"]