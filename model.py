import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

def get_model(num_keypoints, weights_path=None, cpu_only=False):
    
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(1.2, 1.3, 1,4))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu')) if cpu_only else torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model