FROM tensorflow/tensorflow:2.3.0rc0-gpu

RUN apt update -y && \
        apt upgrade -y &&\
        apt install python3 -y && apt install python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install git -y
RUN mkdir /git
WORKDIR /git
RUN git clone https://github.com/hunglc007/tensorflow-yolov4-tflite.git

workdir /git/tensorflow-yolov4-tflite
RUN pip install -r requirements-gpu.txt
