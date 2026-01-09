import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from visionrt import Camera
import torq as tq

model = (
    nn.Sequential(
        nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        nn.Softmax(dim=1),
    )
    .cuda()
    .eval()
)
labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

tq.register(cls=Camera, adapter=lambda x: next(x.stream()))
@tq.register_decorator(lambda cw: cw.print)
class ConsoleWriter:
    def print(self, out1, out2):
        pred_class1 = out1.argmax(dim=1).item()
        pred_class2 = out2.argmax(dim=1).item()
        label1 = labels[pred_class1]
        label2 = labels[pred_class2]
        print(f"{'Camera 1:':<12}{label1:<25} | {'Camera 2:':<12}{label2:<25}")
        
def cam1_preprocess(frame: torch.Tensor):
    return frame.unsqueeze(0)

def cam2_preprocess(frame: cv2.typing.MatLike):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
    return tensor

cam1 = Camera("/dev/video0")
cam2 = cv2.VideoCapture(2)

system = tq.Sequential(
    tq.Concurrent(
        tq.Sequential(cam1, cam1_preprocess, model),
        tq.Sequential(cam2, cam2_preprocess, model)
    ),
    ConsoleWriter()
)

system = tq.compile(system)
system.run()

print(system._graph)

cam1.close()
cam2.release()
