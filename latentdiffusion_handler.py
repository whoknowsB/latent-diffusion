# %%
import argparse, os
import numpy as np
import base64

from io import BytesIO
from PIL import Image

import torch
from torchvision.utils import make_grid
from einops import rearrange

from tqdm import tqdm, trange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from ts.context import Context

# %%
def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

# %%
OPT = argparse.Namespace(
    ddim_steps = 50, # 200
    ddim_eta = 0, # 0
    n_iter = 1, # 1
    W = 256, # 256
    H = 256, # 256
    n_samples = 3, # 4
    scale = 5.0, # 5.0
    plms = True,
    format = 'JPEG'
)

# %%
class ModelHandler(object):

    def __init__(self):
        self.initialized = False
        self.device = None

    def initialize(self, context):

        #  load the model
        self.manifest = context.manifest
        

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']

        model_path = os.path.join(model_dir, serialized_file)
        config_path = os.path.join(model_dir, "config.yaml")

        if not os.path.isfile(model_path):
            raise RuntimeError("Missing the model.pt file")
        if not os.path.isfile(config_path):
            raise RuntimeError("Missing the config.yaml file")

        config = OmegaConf.load(config_path)
        self.model = load_model_from_config(config, model_path, self.device, verbose=True)
        self.sampler = PLMSSampler(self.model) if OPT.plms else DDIMSampler(self.model)
        self.initialized = True

    def handle(self, data, context):
        prompt = str(data[0])
        uc = None
        all_samples=[]

        if OPT.scale != 1.0:
            uc = self.model.get_learned_conditioning(OPT.n_samples * [""])
        for n in trange(OPT.n_iter, desc="Sampling"):
            c = self.model.get_learned_conditioning(OPT.n_samples * [prompt])
            shape = [4, OPT.H//8, OPT.W//8]
            samples_ddim, _ = self.sampler.sample(S=OPT.ddim_steps,
                                                conditioning=c,
                                                batch_size=OPT.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=OPT.scale,
                                                unconditional_conditioning=uc,
                                                eta=OPT.ddim_eta)

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

            #for x_sample in x_samples_ddim:
            #    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            #    x_sample = x_sample.astype(np.uint8)
            #    Image.fromarray(x_sample).save(os.path.join('outputs/samples', f"{base_count:04}.png"))
            #    base_count += 1
            all_samples.append(x_samples_ddim)

        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')

        #grid = make_grid(grid, nrow=OPT.n_samples)
        #grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        #grid = Image.fromarray(grid.astype(np.uint8))
        #buffer = BytesIO()
        #grid.save(buffer, format = OPT.format)
        #encode = base64.b64encode(buffer.getvalue()).decode()
        #return [encode]
        
        images = []
        for image in grid:
            image = 255. * rearrange(image, 'c h w -> h w c').cpu().numpy()
            image = Image.fromarray(image.astype(np.uint8))
            images.append(image)

        encodes = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format = OPT.format)
            encode = base64.b64encode(buffer.getvalue()).decode()
            encodes.append(encode)

        return [encodes]

# %%
#context = Context(
#    'latentdiffusion', 
#    '/home/callmeb/Documents/reply-hackathon-2022/latent-diffusion/models/ldm/text2img-large',
#    {'model':{'serializedFile': 'model.ckpt'}},
#    1,
#    0,
#    'server_version_0'
#    )

# %%
#mh = ModelHandler()
#mh.initialize(context)

# %%
#%%time
#data = [{'body': 'a cat made of wool'}]
#out = mh.handle(data, context)[0]

# %%
#for o in out:
#    im_bytes = base64.b64decode(o.encode())  
#    im_file = BytesIO(im_bytes)
#    img = Image.open(im_file)
#    display(img)

# %%
#torch-model-archiver --model-name latentdiffusion \
#                     --version 1.0 \
#                     --serialized-file /home/callmeb/Documents/reply-hackathon-2022/latent-diffusion/models/ldm/text2img-large/model.ckpt \
#                     --handler latentdiffusion_handler.py \
#                     --extra-files /home/callmeb/Documents/reply-hackathon-2022/latent-diffusion/models/ldm/text2img-large/config.yaml \
#                     --force

# %%
# torchserve --start --models latentdiffusion=latentdiffusion.mar  

# %%
# curl --location --request POST 'http://127.0.0.1:8080/predictions/latentdiffusion?data=a dog with funny blue hat'


