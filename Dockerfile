# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED=TRUE

RUN apt-get update && apt-get install -y libxrender1 python3.9 python3.9-distutils wget git git-lfs \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 && rm -rf /var/lib/apt/lists/*


RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /workspace
COPY requirements_dev.txt .

RUN pip install wheel
# RUN --mount=type=ssh pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements_dev.txt

RUN apt-get update && apt-get install -y build-essential && git clone https://github.com/rlabduke/reduce.git && cd reduce/ && make install && cd ..

COPY . .
RUN python3 -m utils.__init__

ENTRYPOINT ["python3.9"]

# docker build -t pocket-cfdm:latest  .
# docker run -it --rm --gpus all -v '/home/':'/home/' pocket-cfdm:latest -m predict --pdb /home/ubuntu/temp/in.pdb --sdf /home/ubuntu/temp/in.sdf -s /home/ubuntu/temp/out.sdf