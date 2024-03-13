import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class AddSaltPepperNoise(object):

    def __init__(self, density=0, prob=0.5):
        self.density = density
        self.prob = prob
    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        Nd = self.density
        if torch.rand(1) < self.prob:
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = np.repeat(mask, c, axis=2)
            img[mask == 0] = 0
            img[mask == 1] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, prob=0.5):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.prob = prob

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        if torch.rand(1) < self.prob:
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img



class Mycrop(object):

    def __init__(self, left=50, upper=100, right=338, lower=384):
        self.left = left
        self.upper = upper
        self.right = right
        self.lower = lower

    def __call__(self, img):
        image = img.crop([self.left, self.upper, self.right, self.lower])
        image = np.array(image)
        matix = np.zeros_like((100, 100, 3))
        image[184:284, 188:288] = matix
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return image


class Brightness_reduce(object):
    def __init__(self, brightness=120):
        self.brightness = brightness

    def __call__(self, img):
        img_arr = np.array(img.convert('L'))
        if img_arr.mean() > self.brightness:
            brightness_image = transforms.ColorJitter(brightness=(0.5,0.9), contrast=0, saturation=0, hue=0)(img)
        else:
            brightness_image = img

        return brightness_image