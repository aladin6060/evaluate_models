import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import torchvision.transforms as T
import statistics
torch.set_grad_enabled(False)

#Loading the path to all files
files = []
for file in os.listdir("coco/val2017"):
    if file.endswith(".jpg"):
        files.append(os.path.join("coco/val2017", file))

#Initialize the cuda timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#Loading Model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
model.eval()
print("Is Cuda available? {}".format(torch.cuda.is_available()))
#Looping over all images
time = []
for idx,path in enumerate(files):
    im = Image.open(path)
    transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = transform(im).unsqueeze(0)
     # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        img = img.to('cuda')
        model.to('cuda')

    # propagate through the model
    start.record()
    outputs = model(img)
    end.record()
    
    torch.cuda.synchronize()
    time.append(start.elapsed_time(end))
    print("Image {} has been analyzed".format(idx))
mean_time = statistics.median(time)
print("The median time for the analysis of a image is {}".format(mean_time))