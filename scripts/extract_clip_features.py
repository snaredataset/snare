import os
import torch
from PIL import Image
import numpy as np
from numpy import asarray
import clip

import pickle, gzip, json
from tqdm import tqdm


# Set filepaths
shapenet_images_path = './data/shapenet-images/screenshots'
ann_files = ["train.json", "val.json", "test.json"]
folds = './amt/folds_adversarial'

keys = os.listdir(shapenet_images_path)

# Load pre-trained CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Extract CLIP visual features
data = {}
for key in tqdm(keys):
    pngs = os.listdir(os.path.join(shapenet_images_path, f"{key}"))
    pngs = [os.path.join(shapenet_images_path, f"{key}", p) for p in pngs if "png" in p]
    pngs.sort()

    for png in pngs:
        im = Image.open(png)
        image = preprocess(im).unsqueeze(0).to(device)

        image_features = clip_model.encode_image(image).squeeze(0).detach().cpu().numpy()
        image_features = image_features.tolist()
        name = png.split('/')[-1].replace(".png", "")

        data[name] = image_features

save_path = './data/shapenet-clipViT32-frames.json.gz'
json.dump(data, gzip.open(save_path,'wt'))


# Extract CLIP language features
anns = []
for file in ann_files:
    fname_rel = os.path.join(folds, file)
    print(fname_rel)
    with open(fname_rel, 'r') as f:
        anns = anns + json.load(f)

lang_feat = {}
for d in tqdm(anns):
    ann = d['annotation']

    text = clip.tokenize([ann]).to(device)
    feat = clip_model.encode_text(text)

    feat = feat.squeeze(0).detach().cpu().numpy()
    feat = feat.tolist()
    lang_feat[ann] = feat

save_path = './data/langfeat-512-clipViT32.json.gz'
json.dump(lang_feat, gzip.open(save_path,'wt'))