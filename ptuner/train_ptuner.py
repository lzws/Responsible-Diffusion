from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import clip
from PIL import Image
import pandas as pd
import argparse
import os
from tqdm.auto import tqdm
# from tool import plot_curve
from ptuner.ptunerc import ptuner
basepath = '/project/'



safety_concept = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
                                        'sexual, nudity,nude, bodily fluids, blood, obscene gestures, illegal activity, ' \
                                        'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'



def load_components(model_version=' ' ,device='cuda:1'):
    model_version=model_version='CompVis/stable-diffusion-v1-4'

    vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder")

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    return vae, tokenizer, text_encoder, unet, scheduler


def train(device='cuda:1',prompts_path='../prompts/train_prompts/train_i2p_60.csv',save_path='image_output/train_output', image_size=512):

    torch_device = device
    vae, tokenizer, text_encoder, unet, scheduler = load_components(device=device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)



    # direction vector
    p_tuner = ptuner()
    p_tuner.to(device)
    optimizer = optim.Adam(p_tuner.parameters(), lr=0.06)



    # loss
    mse_loss = nn.MSELoss()
    

    df = pd.read_csv(prompts_path)
    
    folder_path = f'{save_path}/{prompts_path.replace(".csv","")}'
    os.makedirs(folder_path, exist_ok=True)


    num_samples = 1
    height = image_size                        # default height of Stable Diffusion
    width = image_size                         # default width of Stable Diffusion
    num_inference_steps = 50           # Number of denoising steps
    guidance_scale = 7.5            # Scale for classifier-free guidance

    for _, row in df.iterrows(): 

        
        prompt = [str(row.prompt)]*num_samples
        unsafe_prompt = [safety_concept]*num_samples



        seed = row.evaluation_seed 
        case_number = row.case_number 

        generator = torch.manual_seed(seed)    # Seed generator to create the inital latent noise

        batch_size = len(prompt)


        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        # prompt embedding [bs, 77, 768]
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
        
        # print(text_embeddings.shape)

        unsafe_input = tokenizer(unsafe_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        # unsafe embedding [bs, 77, 768]
        with torch.no_grad():
            unsafe_embeddings = text_encoder(unsafe_input.input_ids.to(torch_device))[0]
        # print(unsafe_embeddings.shape)

        max_length = text_input.input_ids.shape[-1]

        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]



        # latent code x_t [bs, 4, 64, 64]
        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        # print(latents.shape)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma 
        

        optimizer2 = optim.Adam(unet.parameters(), lr=0.01)
        
        ff = 0

        for t in tqdm(scheduler.timesteps):
            ff += 1
            # print(t)
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            
            if ff >= 10:
                text_embedding_ = p_tuner(text_embeddings) 
            else:
                text_embedding_ = text_embeddings

            text_embeddings_b = torch.cat([uncond_embeddings, text_embedding_])

            embeddings_u = torch.cat([uncond_embeddings, unsafe_embeddings]) 
            
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_b).sample

            # perform guidance

            # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred1 = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            if  ff >= 10 and ff <= 50:

                with torch.no_grad(): # 

                    noise_pred_u = unet(latent_model_input, t, encoder_hidden_states=embeddings_u).sample
                _, noise_pred_unsafe = noise_pred_u.chunk(2)
                # noise_pred_1 = noise_pred_uncond - 7.5 * (noise_pred_unsafe - noise_pred_uncond)
                noise_pred_1 = noise_pred_unsafe + 7.5 * (noise_pred_uncond - noise_pred_unsafe)


                l_m = mse_loss(noise_pred_text, noise_pred_1)

                loss = 10 * l_m 

                optimizer.zero_grad()  
                loss.backward()  
                optimizer2.zero_grad()
                optimizer.step()
            
            # compute the previous noisy sample x_t -> x_t-1 
            latents = scheduler.step(noise_pred1.data, t, latents).prev_sample
            # print(latents.shape)

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")
        



    torch.save(p_tuner.state_dict(), '../models/safedirection.pth')

    
if __name__ == '__main__':
    import time

    start_time = time.time()
    train()

    end_time = time.time()

    execution_time = end_time - start_time


    execution_time_in_hours = execution_time / 3600

  
    print(f"{execution_time} ")
    print(f" {execution_time_in_hours} ")


    
