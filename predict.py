import argparse
from PIL import Image
from pathlib import Path
import numpy as np
import json

import torch
from torch import nn
from torchvision import models

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

parser = argparse.ArgumentParser()

parser.add_argument('input_image', type=Path) 
parser.add_argument('checkpoint', type=Path) 
parser.add_argument('--category-names', default='cat_to_names.json')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--top-k', default=3)

args = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

checkpoint = torch.load(args.checkpoint)

model = getattr(models, checkpoint['arch'])()

classifier = nn.Sequential(
    nn.Linear(checkpoint['input_units'], checkpoint['hidden_units']),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(checkpoint['hidden_units'], checkpoint['output_units']),
    nn.LogSoftmax(dim=1)
)

if 'vgg' in checkpoint['arch']:
    model.classifier = classifier
elif 'resnet' in checkpoint['arch']:
    model.fc = classifier

model.load_state_dict(checkpoint['state_dict'])
model.eval() # disable training

# enable GPU if available and requested
device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu and checkpoint['device']=="cuda") else "cpu")
model.to(device)

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    width, height = 256, 256
    
    if image.width > image.height:
        width = (image.width / image.height) * height
    else:
        height = (image.height / image.width) * width
        
    image.thumbnail((width, height))  # resize shortest size to 256 still preserving aspect ratio
    
    xc, yc = image.width / 2, image.height / 2  # find the centers of resized image
    
    crop_points = (xc - 112, yc - 112, xc + 112, yc + 112)  # +- 112 so we get image of size 224x224 as output
    image_crop = image.crop(crop_points)
    
    image_np = np.array(image_crop, dtype=np.float64)
    image_np /= 255
    
    image_np = (image_np - MEAN) / STD
    
    return torch.tensor(image_np.transpose((2, 0, 1)))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = Image.open(image_path)
    processed_image = process_image(img)
    processed_image = processed_image[None, :].to(device, dtype=torch.float)
    
    preds = torch.exp(model(processed_image))
    
    probs, classes = preds.topk(topk, dim=1)
    # TODO: Implement the code to predict the class from an image file
    return probs[0].tolist(), classes[0].tolist()
    
idx_to_class = {v:k for k, v in checkpoint['class_to_idx'].items()}
probs, classes = predict(args.input_image, model, args.top_k)

names = [cat_to_name[idx_to_class[idx]] for idx in classes]

# top_idx = np.argmax(np.array(probs))
top_idx = 0

print(f"TOP CLASS: {names[top_idx]} {probs[top_idx]*100:.3f}%\n\n")

print(f"Top {args.top_k} preds:")
for i in range(args.top_k):
    print(f"[{i+1}] {probs[i]*100:.3f}% {names[i]}")
