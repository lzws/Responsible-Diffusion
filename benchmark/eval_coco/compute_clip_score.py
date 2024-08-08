import torch
import clip
import numpy as np
from typing import List, Union
from PIL import Image
import random
import pandas as pd

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from diffusers.pipelines import DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from eval_util import get_clip_preprocess


def compute_clip_score(prompts_path, images_path, device='cuda:1'):
    '''
    w (float, optional): The weight of the similarity score. Defaults to 2.5.
    clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
    n_px (int, optional): The size of images. Defaults to 224.
    '''
    n_px = 224
    w = 2.5
    clip_path = '/home/users/zhiwen/project/huggingfacemodels/clip/ViT-B-32.pt'
    model, _ = clip.load(clip_path, device=device)
    image_preprocess, text_preprocess = get_clip_preprocess(
        n_px
    )

    df = pd.read_csv(prompts_path,dtype={'image_id': str})
    clip_score = 0
    start_index = 0
    nums = 0
    for index, row in df.iloc[start_index:].iterrows():
        nums+=1
        image_id = row.image_id
        prompts = str(row.prompt)
        prompts = [prompts]
        image_path = images_path + '/' + str(index)+'_'+str(image_id) + '.png'
        images = [Image.open(image_path)]


        texts_feats = text_preprocess(prompts).to(device=device)
        texts_feats = model.encode_text(texts_feats)


        images_feats = [image_preprocess(img) for img in images]
        images_feats = torch.stack(images_feats, dim=0).to(device=device)
        images_feats = model.encode_image(images_feats)

        # compute the similarity
        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)

        # score = w * images_feats * texts_feats
        score = w * images_feats @ texts_feats.T
        clip_score += score.sum(dim=1).clamp(min=0).cpu().numpy()
    
    mean_score = clip_score / nums

    return mean_score




clip_path = 'huggingfacemodels/clip-vit-large-patch14'

model = CLIPModel.from_pretrained(clip_path).to('cuda:1')
processor = CLIPProcessor.from_pretrained(clip_path)

def get_clip_score2(image_path, text, device):
    image = Image.open(image_path)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
    return logits_per_image

def compute_clip_score2(prompts_path, images_path):
    df = pd.read_csv(prompts_path,dtype={'image_id': str})
    clip_score = 0
    start_index = 0
    indexnum = 0
    for index, row in df.iloc[start_index:].iterrows():
        number = row.image_id
        prompts = str(row.prompt)
        prompts = [prompts]
        image = images_path + '/' + str(indexnum)+'_'+str(number) + '.png'
        result = get_clip_score2(image, prompts, device='cuda:1')
        indexnum += 1
        # print(image)
        # print(index, number, result)
        clip_score += result.item()
    return clip_score / indexnum



if __name__ == '__main__':
    prompts_path = '../prompts/coco_3k.csv'

    data_path = ''
    with torch.no_grad():
        mean_score = compute_clip_score2(prompts_path=prompts_path, images_path=data_path)
    print('mean_score:',mean_score)


