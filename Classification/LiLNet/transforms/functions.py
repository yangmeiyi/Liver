import torch
import torch.nn as nn
import math
import random
from torchvision import transforms
import matplotlib.pyplot as plt

unloader = transforms.ToPILImage()


class SLA_transform:
    def __init__(self, tensor_size, mode='4mask'):
        self.mode = mode
        self.channel = tensor_size[0]
        self.kernel_size = tensor_size[-1]
        self.mask_size = self.kernel_size // 2
        self.mask_copy = torch.ones(tensor_size).cuda()
        if self.mode == '4mask':
            self.mask = [torch.ones(tensor_size).cuda() for i in range(4)]

            self.mask[0][:, 0:self.mask_size, 0:self.mask_size] = 0
            self.mask[1][:, 0:self.mask_size, self.mask_size:self.kernel_size] = 0
            self.mask[2][:, self.mask_size:self.kernel_size, 0:self.mask_size] = 0
            self.mask[3][:, self.mask_size:self.kernel_size, self.mask_size:self.kernel_size] = 0
        elif self.mode == '5mask':
            self.mask = [torch.ones(tensor_size).cuda() for i in range(5)]
            self.mask[0][:, 0:self.mask_size, 0:self.mask_size] = 0
            self.mask[1][:, 0:self.mask_size, self.mask_size:self.kernel_size] = 0
            self.mask[2][:, self.mask_size:self.kernel_size, 0:self.mask_size] = 0
            self.mask[3][:, self.mask_size:self.kernel_size, self.mask_size:self.kernel_size] = 0
            self.mask[4][:, self.mask_size // 2:self.mask_size // 2 + self.mask_size, self.mask_size // 2:self.mask_size // 2 + self.mask_size] = 0
        elif self.mode == 'random_5mask':
            self.mask = [torch.ones(tensor_size).cuda() for i in range(5)]
        elif self.mode == '6mask':
            self.mask = [torch.zeros(tensor_size).cuda() for i in range(6)]
            self.mask[0][:, 0:self.mask_size, 0:self.kernel_size] = 1.
            self.mask[1][:, 0:self.kernel_size, 0:self.mask_size] = 1.
            self.mask[2][:, 0:self.mask_size, 0:self.mask_size] = 1.
            self.mask[2][:, self.mask_size:self.kernel_size, self.mask_size:self.kernel_size] = 1.
            self.mask[3][:, 0:self.mask_size, self.mask_size:self.kernel_size] = 1.
            self.mask[3][:, self.mask_size:self.kernel_size, 0:self.mask_size] = 1.
            self.mask[4][:, self.mask_size:self.kernel_size, 0:self.kernel_size] = 1.
            self.mask[5][:, 0:self.kernel_size, self.mask_size:self.kernel_size] = 1.
        elif self.mode == '9mask':
            self.mask = [torch.ones(tensor_size).cuda() for i in range(9)]
            # dot = [self.kernel_size // 4, self.kernel_size // 2, self.kernel_size - (self.kernel_size // 4)]
            # self.mask[0][:, 0:dot[1], 0:dot[1]] = 0
            # self.mask[1][:, 0:dot[1], dot[0]:dot[2]] = 0
            # self.mask[2][:, 0:dot[1], dot[1]:self.kernel_size] = 0
            # self.mask[3][:, dot[0]:dot[2], 0:dot[1]] = 0
            # self.mask[4][:, dot[0]:dot[2], dot[0]:dot[2]] = 0
            # self.mask[5][:, dot[0]:dot[2], dot[1]:self.kernel_size] = 0
            # self.mask[6][:, dot[1]:self.kernel_size, 0:dot[1]] = 0
            # self.mask[7][:, dot[1]:self.kernel_size, dot[0]:dot[2]] = 0
            # self.mask[8][:, dot[1]:self.kernel_size, dot[1]:self.kernel_size] = 0
            dot = [self.kernel_size // 3, self.kernel_size - self.kernel_size // 3]
            self.mask[0][:, 0:dot[0], 0:dot[0]] = 0
            self.mask[1][:, 0:dot[0], dot[0]:dot[1]] = 0
            self.mask[2][:, 0:dot[0], dot[1]:self.kernel_size] = 0
            self.mask[3][:, dot[0]:dot[1], 0:dot[0]] = 0
            self.mask[4][:, dot[0]:dot[1], dot[0]:dot[1]] = 0
            self.mask[5][:, dot[0]:dot[1], dot[1]:self.kernel_size] = 0
            self.mask[6][:, dot[1]:self.kernel_size, 0:dot[0]] = 0
            self.mask[7][:, dot[1]:self.kernel_size, dot[0]:dot[1]] = 0
            self.mask[8][:, dot[1]:self.kernel_size, dot[1]:self.kernel_size] = 0

    def get_random(self):
        size = 1
        random_x = [random.randint(1, 2), random.randint(5, 6),
                    random.randint(1, 2), random.randint(5, 6),
                    random.randint(3, 4)]
        random_y = [random.randint(1, 2), random.randint(5, 6),
                    random.randint(1, 2), random.randint(5, 6),
                    random.randint(3, 4)]
        for i in range(len(self.mask)):
            self.mask[i][:, random_x[i] - size:random_x[i] + size + 1, random_y[i] - size:random_y[i] + size + 1] = 0

    def mask_trans(self, x):
        bs = x.size()[0]
        x_copy = x * self.mask_copy.unsqueeze(0).repeat(bs, 1, 1, 1).cuda()
        # if 'random' in self.mode:
        #     self.get_random()
        for _mask in self.mask:
            sub_mask = _mask.unsqueeze(0).repeat(bs, 1, 1, 1).cuda()
            x = torch.cat((x, sub_mask * x_copy), 1)
            # print(x_out.size())
        x = x.view(-1, *x_copy.shape[1:])

        # show the pic
        # for i in range(1, len(self.mask) + 2):
        #     image = x[(i - 1) * bs, 0].cpu().clone().squeeze(-1).detach().numpy()
        #     plt.subplot(2, 4, i)
        #     plt.imshow(image, cmap='gray')
        # plt.show()
        return x

    def mask_trans_2fc(self, x):
        bs = x.size()[0]
        x_copy = x * self.mask_copy.unsqueeze(0).repeat(bs, 1, 1, 1).cuda()
        x_mask = x * self.mask[0].unsqueeze(0).repeat(bs, 1, 1, 1).cuda()

        for _mask in self.mask[1:]:
            sub_mask = _mask.unsqueeze(0).repeat(bs, 1, 1, 1).cuda()
            x_mask = torch.cat((x_mask, sub_mask * x_copy), 1)
            # print(x_out.size())
        x_mask = x_mask.view(-1, *x_copy.shape[1:])

        return x_copy, x_mask
