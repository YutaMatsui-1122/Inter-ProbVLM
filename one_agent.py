import os
import argparse

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np
import matplotlib.pyplot as plt

from ProbVLM.src.utils import load_data_loader

from ProbVLM.src.ds import prepare_coco_dataloaders

from ProbVLM.src.networks import *
from ProbVLM.src.train_ProbVLM import *
import clip
from CLIP_prefix_caption.train import *



class OneAgent(nn.Module):
    def __init__(self,):
        super().__init__()
        self.ProbVLM_Net = BayesCap_for_CLIP(inp_dim=512, out_dim=512, hid_dim=256, num_layers=3, p_drop=0.05,)
        self.ProbVLM_Net.load_state_dict(torch.load("ProbVLM/models/ProbVLM_Net_best.pth"))
        self.ProbVLM_Net.eval()
        self.CLIP_Net, self.preprocess = clip.load("ViT-B/32", device='cuda')
        self.CLIP_Net.eval()
        self.prefix_length = 40
        self.prefix_length = 40
        self.prefix_length_clip = 40
        self.prefix_dim = 512
        self.num_layers = 8
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.ClipCap = ClipCaptionPrefix(self.prefix_length, self.prefix_length_clip, self.prefix_dim, self.num_layers,"mlp")
        self.ClipCap.load_state_dict(torch.load('CLIP_prefix_caption/coco_train/coco_prefix_latest.pt'))

    def image_encoder(self,o):
        with torch.no_grad():
            z = self.CLIP_Net.encode_image(o)
            mu_img, alpha_img, sigma_img = self.ProbVLM_Net.img_BayesCap(z)
        return mu_img, alpha_img, sigma_img
    
    def text_encoder(self,t):
        with torch.no_grad():
            z = self.CLIP_Net.encode_text(t)
            mu_cap, alpha_cap, sigma_cap = self.ProbVLM_Net.txt_BayesCap(z)
        return mu_cap, alpha_cap, sigma_cap
    
    def text_decoder(self, z):
        with torch.no_grad():
            z = self.ClipCap.clip_project()
            prefix_embed = self.ClipCap.clip_project(z.float()).reshape(1, self.prefix_length, -1)
            t = generate2(self.ClipCap, self.tokenizer, embed=prefix_embed)
        return t
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    agent = OneAgent()
    agent = agent.cuda()
    agent.eval()


    dataset = 'coco' # coco or flickr

    data_dir = ospj('ProbVLM/dataset/', dataset) # e.g. ospj(expanduser('~'), 'Documents', 'jm', 'data', dataset)
    dataloader_config = mch({
        'batch_size': 8,
        'random_erasing_prob': 0.,
        'traindata_shuffle': True
    })
    from ProbVLM.src.utils import load_data_loader
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    coco_train_loader, coco_valid_loader, coco_test_loader = loaders['train'], loaders['val'], loaders['test']

    test_data = next(iter(coco_test_loader))

    img = test_data[0][0].cuda()
    cap = test_data[1][0].cuda()

    mu_img, alpha_img, sigma_img = agent.image_encoder(img)
    print(mu_img.shape, alpha_img.shape, sigma_img.shape)
    

