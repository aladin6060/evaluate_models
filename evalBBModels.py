import torch
import os
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image
import json

#Importing all models and defining models vector
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
resnet101 = models.resnet101(pretrained=True)
resnet101.eval()
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
resnext50_32x4d.eval()
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
wide_resnet50_2.eval()
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.eval()

from efficientnet_pytorch import EfficientNet
EfficientNetB0 = EfficientNet.from_pretrained('efficientnet-b0')
EfficientNetB0.eval()
EfficientNetB2 = EfficientNet.from_pretrained('efficientnet-b2')
EfficientNetB2.eval()
EfficientNetB4 = EfficientNet.from_pretrained('efficientnet-b4')
EfficientNetB4.eval()
EfficientNetB6 = EfficientNet.from_pretrained('efficientnet-b6')
EfficientNetB6.eval()
models = [resnet18,resnet50,resnet101,resnext50_32x4d,wide_resnet50_2,mobilenet,EfficientNetB0,
          EfficientNetB2,EfficientNetB4,EfficientNetB6]
models_string = ['resnet18','resnet50','resnet101','resnext50_32x4d','wide_resnet50_2','mobilenet','EfficientNetB0',
          'EfficientNetB2','EfficientNetB4','EfficientNetB6']

#load the path for all images that are going to be analyzed
files = []
for file in os.listdir("coco/val2017"):
    if file.endswith(".jpg"):
        files.append(os.path.join("coco/val2017", file))

#Initialize the cuda timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

#Loop over all models and over all 100 images
model_time = []
print("Cuda is available?{}".format(torch.cuda.is_available()))
for model in models:
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
            input_batch = input_batch.to('cuda')
            model.to('cuda')
            

        with torch.no_grad():
            end.record()
            output = model(input_batch)
            start.record()
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        predictions = torch.nn.functional.softmax(output[0], dim=0)
        maxvalue= torch.argmax(predictions)
        torch.cuda.synchronize()
        time.append(end.elapsed_time(start))
    model_time.append(sum(time)/len(time))
    print("Evaluating {}".format(model))

#store results as a dict in a json file
print("writing files")
res = {models_string[i]: model_time[i] for i in range(len(models_string))} 
with open('data.json', 'w') as f:
    json.dump(res, f)
print("finished")