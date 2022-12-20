import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset



class ColorizeDataLoader(Dataset):
    """
    Data loader for the Colorization model
    :param color_path: path to the color images
    :param img_size: size of the image
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :param num_workers: number of workers
    """
    def __init__(self, color_path, limit=None, img_size=224):
        # Raise error if the path is not a directory
        if not os.path.exists(color_path):
            raise Exception("The path is not a exists!")
        # Raise error if images size is not 224
        if img_size != 224:
            raise Exception("The image size must be 224")

        self.limit = limit
        self.color_path = color_path
        self.img_size = img_size
        self.color_channels = 3
        self.gray_channels = 1
        self.data_color = None
        with open(color_path, 'r') as file:
            self.data_color = file.readlines()
        if limit:
            self.data_color = self.data_color[:limit]
        self.size = len(self.data_color)
        
        # Raise error if no data is found
        if len(self.data_color) == 0:
            print(self.color_path)
            raise Exception("Find no images in folder! Check your path", self.color_path)

    def __len__(self):
        return len(self.data_color)
    
    def __getitem__(self, idx):
        # Read the image``
        grey_img, color_img, original_images_shape = self.read_img(idx)

        grey_img = self.transform(grey_img)
        color_img = self.transform(color_img)
        # Return the resized image and the original shape
        return grey_img, color_img, original_images_shape
    def transform(self, img):
        """
        Transform function for the image. It converts the image to tensor
        :param img: image
        :return: tensor image
        """
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
        return trans(img)

    def read_img(self, idx):
        """
        Read and covert the image to the required size
        :param idx: index of the image
        :return grey image, ab image, original image, lab image
        """
        # Read the image
        img_color_path = self.data_color[idx]
        img_color = cv2.imread(img_color_path.strip())
        # Convert ting the image to the required size and convert to lab color space
        lab_img = cv2.cvtColor(
            cv2.resize(img_color, (self.img_size, self.img_size)),
            cv2.COLOR_BGR2Lab)
        # Get original images shape 
        original_shape = img_color.shape
        # print(original_shape)
        return (
            np.reshape(lab_img[:, :, 0], (self.img_size, self.img_size, 1)),
            np.reshape(lab_img[:, :, 1:],(self.img_size, self.img_size,2)),
            original_shape,
            )