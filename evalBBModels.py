import torch
import os
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
import json
import statistics

#Importing all models and defining models vector
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)

from efficientnet_pytorch import EfficientNet
EfficientNetB0 = EfficientNet.from_pretrained('efficientnet-b0')
EfficientNetB2 = EfficientNet.from_pretrained('efficientnet-b2')
EfficientNetB4 = EfficientNet.from_pretrained('efficientnet-b4')
EfficientNetB6 = EfficientNet.from_pretrained('efficientnet-b6')
models = [resnet18,resnet50,resnet101,resnext50_32x4d,wide_resnet50_2,mobilenet,EfficientNetB0,
          EfficientNetB2,EfficientNetB4,EfficientNetB6]
models_string = ['resnet18','resnet50','resnet101','resnext50_32x4d','wide_resnet50_2','mobilenet','EfficientNetB0',
          'EfficientNetB2','EfficientNetB4','EfficientNetB6']

#load the path for all images that are going to be analyzed
files = []
for file in os.listdir("./coco/val2017"):
    if file.endswith(".jpg"):
        files.append(os.path.join("./coco/val2017", file))

#Initialize the cuda timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#Loop over all models and over all 100 images
model_time = []
for model, name_model in zip(models,models_string):
    model.eval()
    time = []
    for path in files:
        input_image = Image.open(path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            img = img.to('cuda')
            model.to('cuda')
        else:
            raise NameError("Cuda is not available")

    
        with torch.no_grad(): 
            start.record()
            output = model(input_batch)
            end.record()
            
        torch.cuda.synchronize()
        time.append(start.elapsed_time(end))
    model_time.append(statistics.median(time))
    print("Evaluating {}".format(name_model))

#store results as a dict in a json file
print("writing files")
res = {models_string[i]: model_time[i] for i in range(len(models_string))} 
with open('data.json', 'w') as f:
    json.dump(res, f)
print("finished")