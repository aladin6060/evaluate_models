import torch
import os
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as T
from PIL import Image
import json
import statistics

#Initialize the cuda timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#read the path for all images
files = []
for file in os.listdir("coco/val2017"):
    if file.endswith(".jpg"):
        files.append(os.path.join("coco/val2017", file))
#Load model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval();

#extract layers
transformer = model.transformer
class_embed = model.class_embed
bbox_embed = model.bbox_embed
query_embed = model.query_embed
input_proj = model.input_proj
backbone = model.backbone

time_backbone = []
time_transformer = []
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
    else:
        raise NameError("Cuda is not available")


    # propagate through the model
    # timing the backbone
    start.record()
    out1 = backbone(img)
    end.record()
    torch.cuda.synchronize()
    time_backbone.append(start.elapsed_time(end))

    out2 = input_proj(out1)

    out3 = query_embed(out2)

    out4 = bbox_embed(out3)

    out5 = class_embed(out4)
    #timing the transformer
    start.record()
    out = transformer(out5)
    end.record()
    torch.cuda.synchronize()
    time_transformer.append(start.elapsed_time(end))

    print("Image {} has been analyzed".format(idx))
backbone_median = statistics.median(time_backbone)
transformer_median = statistics.median(time_transformer)