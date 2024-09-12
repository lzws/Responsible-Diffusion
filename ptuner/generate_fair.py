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
import math
# from tool import plot_curve
from ptuner.ptunerc import ptuner
import random

basepath = '/project/'



# safety_concept = 'hate, harassment, violence, suffering, humiliation, harm, suicide, ' \
#                                         'sexual, nudity,nude, bodily fluids, blood, obscene gestures, illegal activity, ' \
#                                         'drug use, theft, vandalism, weapons, child abuse, brutality, cruelty'



def load_components(model_version=' ' ,device='cuda:1'):
    model_version='CompVis/stable-diffusion-v1-4'
    
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




def latents2images(latents,vae):
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images    

def generate_images(device='cuda:1',prompts_path='prompts/test_1.csv',fair_type='gender_fair',save_path='fair_output', 
                    start_perturb_step=20,end_perturb_step=50, start_tuning_step=150, end_tuning_step=20, tdown=600,  file_name='no-tuning-beta-tdown600-20-50', safety_concept = 'sexual, nudity,nude,', image_size=512):

    torch_device = device
    vae, tokenizer, text_encoder, unet, scheduler = load_components(device=device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)



    # loss
    mse_loss = nn.MSELoss()
    
    df = pd.read_csv(prompts_path)
    prompts_name = prompts_path.split('/')[-1]
    folder_path = f'{save_path}/{file_name}/{fair_type}/{prompts_name.replace(".csv","")}'
    os.makedirs(folder_path, exist_ok=True)

    # 
    num_samples = 1
    height = image_size                        # default height of Stable Diffusion
    width = image_size                         # default width of Stable Diffusion
    num_inference_steps = 50           # Number of denoising steps
    guidance_scale = 7.5            # Scale for classifier-free guidance

    tuners = []

    if fair_type == 'gender_fair':
        
        # male direction
        male_tuner = ptuner()
        male_tuner.load_state_dict(torch.load('models/fair_models/train_male_p.pth'))
        male_tuner.to(device)
        tuners.append(male_tuner)

        female_tuner = ptuner()
        female_tuner.load_state_dict(torch.load('models//fair_models/train_female_p.pth'))
        female_tuner.to(device)
        tuners.append(female_tuner)
    if fair_type == 'race_fair':
        white_tuner = ptuner()
        white_tuner.load_state_dict(torch.load('models/train_white_50.pth'))
        white_tuner.to(device)
        tuners.append(white_tuner)

        black_tuner = ptuner()
        black_tuner.load_state_dict(torch.load('models/train_black_50.pth'))
        black_tuner.to(device)
        tuners.append(black_tuner)
        
        Asian_tuner = ptuner()
        Asian_tuner.load_state_dict(torch.load('models/train_Asian.pth'))
        Asian_tuner.to(device)
        tuners.append(Asian_tuner)

    # optimizer = optim.Adam(p_tuner.parameters(), lr=0.06)
    hh = 0
    for _, row in df.iterrows(): # 

        

        prompt = [str(row.prompt)]*num_samples
        unsafe_prompt = [safety_concept]*num_samples

        seed = row.evaluation_seed 
        case_number = row.case_number 

        generator = torch.manual_seed(seed)    # Seed generator to create the inital latent noise
        # random.seed(seed)

        fair_num = random.randint(0, len(tuners)-1)

        

        batch_size = len(prompt)


        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        # prompt embedding  [bs, 77, 768]
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        
        
        # print(text_embeddings.shape)

        unsafe_input = tokenizer(unsafe_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        # unsafe embedding  [bs, 77, 768]
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
        

        

        step = 0


        for t in tqdm(scheduler.timesteps):
            step += 1
            # print(t)
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.    
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            
            if step >= start_perturb_step and step <= end_perturb_step:

                text_embedding_ = tuners[fair_num](text_embeddings,1) 
            else:
                text_embedding_ = text_embeddings 
           
            text_embeddings_b = torch.cat([uncond_embeddings, text_embedding_])

            embeddings_u = torch.cat([uncond_embeddings, unsafe_embeddings]) 
            
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_b).sample

            # perform guidance

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            
            
            # compute the previous noisy sample x_t -> x_t-1 
            latents = scheduler.step(noise_pred.data, t, latents).prev_sample
            # print(latents.shape)

        pil_images = latents2images(latents,vae)

        for num, im in enumerate(pil_images):

            im.save(f"{folder_path}/{case_number}_{num}.png")



    
if __name__ == '__main__':
    import time

    start_time = time.time()

    device='cuda:0'

    prompts_path='prompts/doctor.csv'


    # model_name='train_Asian.pth'
    fair_type='gender_fair'
    
    save_path='fair_output'

    start_perturb_step=10
    end_perturb_step=50

    start_tuning_step=150
    end_tuning_step=20

    
    file_name='no-tuning-beta-1-10-50'
    safety_concept = 'sexual, nudity,nude,'

    with torch.no_grad():
        generate_images(device=device, prompts_path=prompts_path,fair_type=fair_type,save_path=save_path,start_perturb_step=start_perturb_step,end_perturb_step=end_perturb_step,tdown=tdown,file_name=file_name,safety_concept=safety_concept)

   
    end_time = time.time()
    
    execution_time = end_time - start_time

    
    execution_time_in_hours = execution_time / 3600

    
    print(f" {execution_time} ")
    print(f" {execution_time_in_hours} ")


    
