'''
sdd.py: Stable Diffusion daemon. Pre-load the model and serve image prompts via FastAPI.

This fetches SD from Hugging Face, so huggingface-cli login first.

Reduces rendering time is about 10 it/s (~4 seconds for 40 steps) for 512x512 on an RTX 2080.

'''
import random
import gc

from threading import Lock
from typing import Optional
from io import BytesIO

import torch

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse

from transformers import CLIPTextModel, CLIPTokenizer, AutoFeatureExtractor, logging
# from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from tqdm.auto import tqdm
from torch import autocast
from PIL import Image

app = FastAPI()

MODELS = {
    "pipeline": {
        "name": "stabilityai/stable-diffusion-2",
        "sub": ""
    },
    "safety": {
        "name": "CompVis/stable-diffusion-safety-checker",
        "sub": ""
    }
}

# One lock for each available GPU (only one supported for now)
GPUS = {}
for i in range(torch.cuda.device_count()):
    GPUS[i] = Lock()

if not GPUS:
    raise RuntimeError("No GPUs detected. Check your config and try again.")

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# The CompVis safety model.
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(MODELS["safety"]["name"], subfolder=MODELS["safety"]["sub"])
safety_checker = StableDiffusionSafetyChecker.from_pretrained(MODELS["safety"]["name"], subfolder=MODELS["safety"]["sub"])

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(MODELS["pipeline"]["name"], subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(MODELS["pipeline"]["name"], scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def naughty(image):
    ''' Returns True if naughty bits are detected, else False. '''
    safety_checker_input = safety_feature_extractor([image], return_tensors="pt")
    _, has_nsfw_concept = safety_checker(images=[image], clip_input=safety_checker_input.pixel_values)
    return has_nsfw_concept[0]

def wait_for_gpu():
    ''' Return the device name of first available GPU. Blocks until one is available and sets the lock. '''
    while True:
        gpu = random.choice(list(GPUS))
        if GPUS[gpu].acquire(timeout=0.5):
            return gpu

def clear_cuda_mem():
    ''' Try to recover from CUDA OOM '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception as e:
            pass

    gc.collect()
    torch.cuda.empty_cache()

def generate_image(prompt, seed, steps, width=768, height=768, guidance=15):
    ''' Generate and return an image array using the first available GPU '''
    gpu = wait_for_gpu()

    try:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        return pipe(
            prompt,
            negative_prompt="meme youtube 'play button' 'computer graphics' caption",
            generator=generator,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance_scale=guidance
        ).images[0]

    except RuntimeError:
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        raise HTTPException(
            status_code=507,
            detail="Out of CUDA memory. Try smaller values for width and height."
        )

    finally:
        clear_cuda_mem()
        GPUS[gpu].release()

def safe_generate_image(prompt, seed, steps, width=768, height=768, guidance=15, safe=True):
    ''' Generate an image and check NSFW. Returns a FastAPI StreamingResponse. '''

    image = generate_image(prompt, seed, steps, width, height, guidance)

    if safe and naughty(image):
        print("🍆 detected!!!1!")
        prompt = "An adorable teddy bear running through a grassy field, early morning volumetric lighting"
        image = generate_image(prompt, seed, steps, width, height, guidance)

    # Set the EXIF data. See PIL.ExifTags.TAGS to map numbers to names.
    exif = image.getexif()
    exif[271] = prompt # exif: Make
    exif[272] = MODELS["pipeline"]["name"] # exif: Model
    exif[305] = f'seed={seed}, steps={steps}' # exif: Software

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85, exif=exif)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg", headers={
        'Content-Disposition': 'inline; filename="synthesis.jpg"'}
    )

@app.get("/", status_code=302)
async def root():
    ''' Hi there! '''
    return RedirectResponse("/docs")

@app.post("/generate/")
def generate(
    prompt: Optional[str] = Query(""),
    seed: Optional[int] = Query(-1),
    steps: Optional[int] = Query(ge=1, le=100, default=40),
    width: Optional[int] = Query(768),
    height: Optional[int] = Query(768),
    guidance: Optional[float] = Query(15),
    safe: Optional[bool] = Query(True),
    ):
    ''' Generate an image with Stable Diffusion '''

    if seed < 0:
        seed = random.randint(0,2**64-1)

    prompt = prompt.strip().replace('\n', ' ')

    torch.cuda.empty_cache()

    return safe_generate_image(prompt, seed, steps, width, height, guidance, safe)
