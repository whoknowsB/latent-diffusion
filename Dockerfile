FROM pytorch/torchserve:latest-gpu

COPY . /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ndefault_workers_per_model=2" >> /home/model-server/config.properties

RUN apt-get install -y wget
RUN if [ ! -f /home/model-server/models/ldm/text2img-large/model.ckpt ]; then \
        echo "Checkpoint not found!" && \
        wget -O /home/model-server/models/ldm/text2img-large/model.ckpt https://storage.googleapis.com/latentdiffusion-bucket/models/ldm/text2img-large/model.ckpt ; \
    fi 

RUN pip install -r /home/model-server/requirements.txt

RUN pip install -e "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" -e "git+https://github.com/openai/CLIP.git@main#egg=clip"
RUN pip install -e /home/model-server/

USER model-server

RUN torch-model-archiver \
    --model-name=latentdiffusion \
    --version=1.0 \
    --serialized-file /home/model-server/models/ldm/text2img-large/model.ckpt \
    --handler=/home/model-server/latentdiffusion_handler.py \
    --extra-files=/home/model-server/models/ldm/text2img-large/config.yaml \
    --export-path=/home/model-server/model-store 

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "latentdiffusion=latentdiffusion.mar"]