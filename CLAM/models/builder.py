import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import pandas as pd
import torch.nn as nn
import torchvision
from huggingface_hub import login

login("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")

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def vit_small(pretrained, progress, key, **kwargs):
    from timm.models.vision_transformer import VisionTransformer
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0
    )
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def load_resnet18(device='cuda'):
    # Load the ResNet-18 architecture without pretrained weights
    model = torchvision.models.resnet18(pretrained=False)
    state = torch.load('tenpercent_resnet18.ckpt', map_location=device)
    state_dict = {k.replace('model.', '').replace('resnet.', ''): v for k, v in state['state_dict'].items()}
    # Load weights into the model
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
    # Modify the final layer based on the return_preactivation flag
    model.fc = torch.nn.Identity()
    # Move the model to the specified device
    model = model.to(device)
    return model

    return model
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet':
        model = TimmCNNEncoder()
    elif model_name == 'uni':
        model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif model_name == 'optimus1':
        model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    elif model_name == 'uni2':
        timm_kwargs = {'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24, 'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
            'num_classes': 0, 'no_embed_class': True, 'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True}
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        img_transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    # print(model)
    if model_name not in ['uni2','uni']:
        constants = MODEL2CONSTANTS[model_name]
        img_transforms = get_eval_transforms(mean=constants['mean'],
                                            std=constants['std'],
                                            target_img_size = target_img_size)
    return model, img_transforms

