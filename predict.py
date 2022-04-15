import numpy as np
import monkey_patch_torch
import torch
import os
from torch.utils.data import DataLoader
import torchvision
from object_detection.utils import collate_fn
from settings import test_dataset_path, keypoints_classes_ids2names, weights_path, score_threshold
from class_dataset import ClassDataset
from model import get_model
from image_processing import visualize

def predict(images, device):
    images = list(image.to(device) for image in images)
    with torch.no_grad():
        model = get_model(num_keypoints = len(keypoints_classes_ids2names), weights_path=weights_path, cpu_only=True)
        model.to(device)
        model.eval()
        output = model(images)

    print("Predictions: \n", output)

    image = (images[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
    scores = output[0]['scores'].detach().cpu().numpy()

    high_scores_idxs = np.where(scores > score_threshold)[0].tolist() # Indexes of boxes with scores > 0.7
    high_scores_idxs = [high_scores_idxs[0]] # only best one
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append([list(map(int, kp[:2])) for kp in kps])

    bboxes = []
    for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        bboxes.append(list(map(int, bbox.tolist())))
        
    return visualize(image, bboxes, keypoints)

def main():
    dataset_test = ClassDataset(test_dataset_path, transform=None, demo=False)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, collate_fn=collate_fn)
    iterator = iter(data_loader_test)
    images, targets = next(iterator)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    i = 1

    os.makedirs("./out", exist_ok=True)

    for images, _ in iterator:
        fig = predict(images, device)
        fig.savefig("./out/predict_{}.png".format(i))
        i += 1

if __name__ == "__main__":
    main()