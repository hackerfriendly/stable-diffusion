'''
sdd.py: Stable Diffusion daemon

Pre-load the model and serve image prompts via FastAPI.

Reduces rendering time from 1:20 to about 3.5 seconds on an RTX 2080.
'''
import random
import uuid
import tempfile

from pathlib import Path
from threading import Lock
from typing import Optional
from io import BytesIO

import torch
import exif

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image

app = FastAPI()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True)

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True)

# The noise scheduler
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

def generate_image(prompt, seed, steps, width=512, height=512, guidance=7.5):
    # Some settings
    generator = torch.manual_seed(seed)
    batch_size = 1

    # Prep text
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(steps)

    # Prep latents
    latents = torch.randn(
      (batch_size, unet.in_channels, height // 8, width // 8),
      generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            # Scale the latents (preconditioning):
            # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5) # Diffusers 0.3 and below
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, i, latents)["prev_sample"] # Diffusers 0.3 and below
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Display
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    buf = BytesIO()
    pil_images[0].save(buf, format="JPEG", quality=100)

    buf.seek(0)
    meta = exif.Image(buf)
    meta.model = 'Stable Diffusion v1.4'
    meta.software = f'seed={seed}, steps={steps}'
    meta.make = prompt

    with open('out.jpg', 'wb') as f:
        f.write(meta.get_file())

    buf.write(meta.get_file())
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg", headers={
        'Content-Disposition': 'inline; filename="out.jpg"'}
    )

@app.get("/")
async def root():
    ''' Hi there! '''
    return {"message": "Stable Diffusion server. Try /docs"}

@app.post("/generate/")
async def generate(
    prompt: Optional[str] = Query(""),
    seed: Optional[int] = Query(-1),
    steps: Optional[int] = Query(30),
    width: Optional[int] = Query(512),
    height: Optional[int] = Query(512),
    ):
    ''' Make an image and return it '''

    if width * height > 287744:
        raise HTTPException(
            status_code=422,
            detail='Out of GPU memory. Total width * height must be < 287744 pixels.'
        )


    if seed < 0:
        seed = random.randint(0,2**64-1)

    prompt = prompt.strip().replace('\n', ' ')

    return generate_image(prompt, seed, steps, width, height)
