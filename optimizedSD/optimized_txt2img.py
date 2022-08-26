import argparse, os, sys, glob, random
import torch
import numpy as np
import copy
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


config = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
device = "cuda"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=5,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--small_batch",
    action='store_true',
    help="Reduce inference time when generate a smaller batch of images",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
opt = parser.parse_args()

tic = time.time()
os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir

sample_path = os.path.join(outpath, "_".join(opt.prompt.split()))[:255]
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
grid_count = len(os.listdir(outpath)) - 1
seed_everything(opt.seed)

sd = load_model_from_config(f"{ckpt}")
li = []
lo = []
for key, value in sd.items():
    sp = key.split('.')
    if(sp[0]) == 'model':
        if('input_blocks' in sp):
            li.append(key)
        elif('middle_block' in sp):
            li.append(key)
        elif('time_embed' in sp):
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd['model1.' + key[6:]] = sd.pop(key)
for key in lo:
    sd['model2.' + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")
config.modelUNet.params.ddim_steps = opt.ddim_steps

if opt.small_batch:
    config.modelUNet.params.small_batch = True
else:
    config.modelUNet.params.small_batch = False



model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
    
modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
    
modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()

if opt.precision == "autocast":
    model.half()
    modelCS.half()

start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))


precision_scope = autocast if opt.precision=="autocast" else nullcontext
with torch.no_grad():

    all_samples = list()
    for n in trange(opt.n_iter, desc="Sampling"):
        for prompts in tqdm(data, desc="data"):
             with precision_scope("cuda"):
                modelCS.to(device)
                uc = None
                if opt.scale != 1.0:
                    uc = modelCS.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                
                c = modelCS.get_learned_conditioning(prompts)
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                mem = torch.cuda.memory_allocated()/1e6
                modelCS.to("cpu")
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)


                samples_ddim = model.sample(S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=start_code)

                modelFS.to(device)
                print("saving images")
                for i in range(batch_size):
                    
                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    # do the safety dance!
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                    # substitute
                    x_sample = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                # for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1


                mem = torch.cuda.memory_allocated()/1e6
                modelFS.to("cpu")
                while(torch.cuda.memory_allocated()/1e6 >= mem):
                    time.sleep(1)

                # if not opt.skip_grid:
                #     all_samples.append(x_samples_ddim)
                del samples_ddim
                print("memory_final = ", torch.cuda.memory_allocated()/1e6)

        # if not skip_grid:
        #     # additionally, save as grid
        #     grid = torch.stack(all_samples, 0)
        #     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        #     grid = make_grid(grid, nrow=n_rows)

        #     # to image
        #     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        #     Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        #     grid_count += 1

toc = time.time()

time_taken = (toc-tic)/60.0

print(("Your samples are ready in {0:.2f} minutes and waiting for you here \n" + sample_path).format(time_taken))