
import torch
import clip
import numpy as np
from typing import List, Union
from PIL import Image
import random

# from src.engine.train_util import text2img
# from src.configs.config import RootConfig
# from src.misc.clip_templates import imagenet_templates

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from diffusers.pipelines import DiffusionPipeline

def get_clip_preprocess(n_px=224):
    def Convert(image):
        return image.convert("RGB")

    image_preprocess = Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def text_preprocess(text):
        return clip.tokenize(text, truncate=True)

    return image_preprocess, text_preprocess


@torch.no_grad()
def clip_score(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image, str]],
    texts: str,
    w: float = 2.5,
    clip_model: str = "ViT-B/32",
    n_px: int = 224,
    cross_matching: bool = False,
):
    """
    Compute CLIPScore (https://arxiv.org/abs/2104.08718) for generated images according to their prompts.
    *Important*: same as the official implementation, we take *SUM* of the similarity scores across all the
        reference texts. If you are evaluating on the Concept Erasing task, it might should be modified to *MEAN*,
        or only one reference text should be given.

    Args:
        images (List[Union[torch.Tensor, np.ndarray, PIL.Image.Image, str]]): A list of generated images.
            Can be a list of torch.Tensor, numpy.ndarray, PIL.Image.Image, or a str of image path.
        texts (str): A list of prompts.
        w (float, optional): The weight of the similarity score. Defaults to 2.5.
        clip_model (str, optional): The name of CLIP model. Defaults to "ViT-B/32".
        n_px (int, optional): The size of images. Defaults to 224.
        cross_matching (bool, optional): Whether to compute the similarity between images and texts in cross-matching manner.

    Returns:
        score (np.ndarray): The CLIPScore of generated images.
            size: (len(images), )
    """
    if isinstance(texts, str):
        texts = [texts]
    if not cross_matching:
        assert len(images) == len(
            texts
        ), "The length of images and texts should be the same if cross_matching is False."

    if isinstance(images[0], str):
        images = [Image.open(img) for img in images]
    elif isinstance(images[0], np.ndarray):
        images = [Image.fromarray(img) for img in images]
    elif isinstance(images[0], torch.Tensor):
        images = [Image.fromarray(img.cpu().numpy()) for img in images]
    else:
        assert isinstance(images[0], Image.Image), "Invalid image type."

    model, _ = clip.load(clip_model, device="cuda")
    image_preprocess, text_preprocess = get_clip_preprocess(
        n_px
    )  # following the official implementation, rather than using the default CLIP preprocess

    # extract all texts
    texts_feats = text_preprocess(texts).cuda()
    texts_feats = model.encode_text(texts_feats)

    # extract all images
    images_feats = [image_preprocess(img) for img in images]
    images_feats = torch.stack(images_feats, dim=0).cuda()
    images_feats = model.encode_image(images_feats)

    # compute the similarity
    images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
    texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
    if cross_matching:
        score = w * images_feats @ texts_feats.T
        # TODO: the *SUM* here remains to be verified
        return score.sum(dim=1).clamp(min=0).cpu().numpy()
    else:
        score = w * images_feats * texts_feats
        return score.sum(dim=1).clamp(min=0).cpu().numpy()