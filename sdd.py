'''
sdd.py: Stable Diffusion daemon. Pre-load the model and serve image prompts via FastAPI.

This fetches SD from Hugging Face, so huggingface-cli login first.

Reduces rendering time to about 8 it/s (~5 seconds for 40 steps) for 512x512 on an RTX 2080.
'''
import random
import gc

from threading import Lock
from typing import Optional
from io import BytesIO

import torch

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse

from transformers import CLIPTextModel, CLIPTokenizer, AutoFeatureExtractor, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from tqdm.auto import tqdm
from torch import autocast
from PIL import Image

app = FastAPI()

MODELS = {
    "unet": {
        # "name" = "CompVis/stable-diffusion-v1-4",
        "name": "runwayml/stable-diffusion-v1-5",
        "sub": "unet"
    },
    "vae": {
        "name": "stabilityai/sd-vae-ft-ema",
        "sub": ""
    },
    "tokenizer": {
        "name": "openai/clip-vit-large-patch14",
        "sub": ""
    },
    "text_encoder": {
        "name": "openai/clip-vit-large-patch14",
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

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(MODELS["vae"]["name"], subfolder=MODELS["vae"]["sub"], use_auth_token=True).half()

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained(MODELS["tokenizer"]["name"], subfolder=MODELS["tokenizer"]["sub"])
text_encoder = CLIPTextModel.from_pretrained(MODELS["text_encoder"]["name"], subfolder=MODELS["text_encoder"]["sub"])

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(MODELS["unet"]["name"], subfolder=MODELS["unet"]["sub"], use_auth_token=True).half()

# The CompVis safety model.
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(MODELS["safety"]["name"], subfolder=MODELS["safety"]["sub"])
safety_checker = StableDiffusionSafetyChecker.from_pretrained(MODELS["safety"]["name"], subfolder=MODELS["safety"]["sub"])

# The noise scheduler
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)

# To the GPU we go!
vae = vae.to('cuda')
text_encoder = text_encoder.to('cuda')
unet = unet.to('cuda')

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

def generate_image(prompt, seed, steps, width=512, height=512, guidance=7.5):
    ''' Generate and return an image array using the first available GPU '''
    gpu = wait_for_gpu()

    try:
        # Prep text
        text_input = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to('cuda'))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to('cuda'))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) # pylint: disable=no-member
        # Prep Scheduler
        scheduler.set_timesteps(steps)

        # Prep latents
        latents = torch.randn( # pylint: disable=no-member
          (1, unet.in_channels, height // 8, width // 8),
          generator=torch.manual_seed(seed),
        ).half()
        latents = latents.to('cuda')
        latents = latents * scheduler.init_noise_sigma

        # Loop
        with autocast("cuda"):
            for _, ts in tqdm(enumerate(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = scheduler.scale_model_input(torch.cat([latents] * 2), ts) # pylint: disable=no-member

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, ts, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, ts, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # Display
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")

        return images[0]

    except RuntimeError:
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        raise HTTPException(
            status_code=507,
            detail="Out of CUDA memory. Try smaller values for width and height."
        )

    finally:
        clear_cuda_mem()
        GPUS[gpu].release()

def safe_generate_image(prompt, seed, steps, width=512, height=512, guidance=7.5, safe=True):
    ''' Generate an image and check NSFW. Returns a FastAPI StreamingResponse. '''

    image = generate_image(prompt, seed, steps, width, height, guidance)

    if safe and naughty(image):
        print("🍆 detected!!!1!")
        prompt = "An adorable teddy bear running through a grassy field, early morning volumetric lighting"
        image = generate_image(prompt, seed, steps, width, height, guidance)

    out = Image.fromarray(image)

    # Set the EXIF data. See PIL.ExifTags.TAGS to map numbers to names.
    exif = out.getexif()
    exif[271] = prompt # exif: Make
    exif[272] = MODELS["unet"]["name"] # exif: Model
    exif[305] = f'seed={seed}, steps={steps}' # exif: Software

    buf = BytesIO()
    out.save(buf, format="JPEG", quality=85, exif=exif)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg", headers={
        'Content-Disposition': 'inline; filename="synthesis.jpg"'}
    )

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Stable Diffusion server. Try /docs"}

@app.post("/generate/")
async def generate(
    prompt: Optional[str] = Query(""),
    seed: Optional[int] = Query(-1),
    steps: Optional[int] = Query(ge=1, le=100, default=40),
    width: Optional[int] = Query(512),
    height: Optional[int] = Query(512),
    guidance: Optional[float] = Query(7.5),
    safe: Optional[bool] = Query(True),
    ):
    ''' Generate an image with Stable Diffusion '''

    if seed < 0:
        seed = random.randint(0,2**64-1)

    prompt = prompt.strip().replace('\n', ' ')

    torch.cuda.empty_cache()

    return safe_generate_image(prompt, seed, steps, width, height, guidance, safe)
