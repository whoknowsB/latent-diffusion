FROM pytorch/torchserve:latest-gpu

USER root
RUN apt-get install -y wget

COPY ./latentdiffusion_handler.py /home/model-server
ADD ./setup.py /home/model-server/ldm_package
COPY ./configs/ /home/model-server/ldm_package/configs/
COPY ./models/ /home/model-server/ldm_package/models/
COPY ./ldm/ /home/model-server/ldm_package/ldm/
RUN pip install -e /home/model-server/ldm_package


RUN if [ ! -f /home/model-server/models/ldm/text2img-large/model.ckpt ]; then \
        echo "Checkpoint not found!" \
        wget -O /home/model-server/models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt \
    fi 

ADD ./requirements.txt /home/model-server
RUN pip install -r /home/model-server/requirements.txt

USER model-server

RUN torch-model-archiver \
    --model-name=latentdiffusion \
    --version=1.0 \
    --serialized-file /home/model-server/models/ldm/text2img-large/model.ckpt \
    --handler=/home/model-server/latentdiffusion_handler.py \
    --extra-files=/home/model-server/models/ldm/text2img-large/config.yaml 

CMD ["torchserve", \
     "--start", \
#     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "latentdiffusion=latentdiffusion.mar"]