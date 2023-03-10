import torch
from colorization import Colorization
from colorize_dataloader import ColorizeDataLoader
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import os
import config as config
from chromagan_utils import *
import cv2
from tqdm import tqdm
# load the Colorization model from weights
def load_model(model_path):
    model = Colorization().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Colorize the images using model
def colorize(model, images_path):
    test_dataloader = ColorizeDataLoader(images_path)
    test_dataloader = DataLoader(
        test_dataloader, batch_size=4,
        shuffle=True, num_workers=2)
    print('sampling images')
    for idx, (gray, _, original_image_size) in enumerate(tqdm(test_dataloader)):
        l_3 = torch.cat([gray, gray, gray], dim=1).to(config.DEVICE)
        colored, _ = model(l_3)

        gray = gray.detach().cpu().numpy()
        colored = colored.detach().cpu().numpy()

        if not os.path.exists(config.OUTPUT_PATH):
            os.makedirs(config.OUTPUT_PATH)
            
        for i in range(4):
            original_result = reconstruct(deprocess(gray)[i], deprocess(colored)[i])
            original_result = cv2.resize(
                original_result, 
                (int(original_image_size[1][i]), int(original_image_size[0][i])),
                interpolation = cv2.INTER_LANCZOS4)
            cv2.imwrite(config.OUTPUT_PATH + str(idx) + '_' + str(i) + '.png', original_result)

def main():
    model = load_model(config.MODEL_PATH)
    path = 'datasets/Gray'
    colorize(model, path)

if __name__ == '__main__':
    ...
#     main()