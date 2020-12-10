import torch
import os
import torch
import torchvision
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as T
from PIL import Image
import statistics
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

#Initialize the cuda timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#read the path for all images
files = []
for file in os.listdir("./coco/val2017"):
    if file.endswith(".jpg"):
        files.append(os.path.join("./coco/val2017", file))
print("There are {} images to analyze".format(len(files)))
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
    im = Image.open(path).convert('RGB')
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

    ###
    if isinstance(img, (list, torch.Tensor)):
      img = nested_tensor_from_tensor_list(img)
    

    # propagate through the model
    # timing the backbone
    start.record()
    features, pos = backbone(img)
    end.record()
    torch.cuda.synchronize()
    time_backbone.append(start.elapsed_time(end))

    src, mask = features[-1].decompose()

    assert mask is not None
    #timing the transformer
    start.record()
    hs = transformer(input_proj(src), mask, query_embed.weight, pos[-1])[0]
    end.record()
    torch.cuda.synchronize()
    time_transformer.append(start.elapsed_time(end))
    
    outputs_class = class_embed(hs)
    outputs_coord = bbox_embed(hs).sigmoid()

    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
    

    print("Image {} has been analyzed".format(idx))
backbone_median = statistics.median(time_backbone)
transformer_median = statistics.median(time_transformer)
print("The median evaluation time for the backbone is {} ms".format(backbone_median))
print("The median evaluation time for the transformer is {} ms".format(transformer_median))
