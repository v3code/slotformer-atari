FROM condaforge/miniforge3
WORKDIR /code/
RUN apt-get update
RUN apt-get install zip build-essentials ffmpeg libsm6 libxext6 wget
COPY . .
RUN conda env create -f ./environment.yml
RUN conda init
ENTRYPOINT ["tail", "-f", "/dev/null"]