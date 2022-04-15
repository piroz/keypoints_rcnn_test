import numpy as np
import os
import monkey_patch_torch

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F



import object_detection.utils, object_detection.engine, object_detection.train
from object_detection.utils import collate_fn
from object_detection.engine import train_one_epoch, evaluate
from class_dataset import ClassDataset
from model import get_model
from image_processing import visualize, train_transform

from settings import train_dataset_path, test_dataset_path, num_epochs, keypoints_classes_ids2names, weights_path

def train():
    dataset = ClassDataset(train_dataset_path, transform=train_transform(), demo=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    iterator = iter(data_loader)
    batch = next(iterator)

    print("Original targets:\n", batch[3], "\n\n")
    print("Transformed targets:\n", batch[1])

    image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

    keypoints = []
    for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints.append([kp[:2] for kp in kps])

    image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

    keypoints_original = []

    for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
        keypoints_original.append([kp[:2] for kp in kps])

    fig = visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)

    os.makedirs("./out", exist_ok=True)
    fig.savefig("./out/train_dataset.png")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_train = ClassDataset(train_dataset_path, transform=train_transform(), demo=False)
    dataset_test = ClassDataset(test_dataset_path, transform=None, demo=False)

    data_loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_keypoints = len(keypoints_classes_ids2names))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device)
        
    # Save model weights after training
    torch.save(model.state_dict(), weights_path)

if __name__ == "__main__":
    train()