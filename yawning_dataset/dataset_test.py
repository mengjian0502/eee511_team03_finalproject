"""
test the saved dataset
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    dataset = torch.load('./yawnDD_image.pt')
    target = torch.load('./yawnDD_label.pt')
    print(target)
    for kk in range(len(target)):
        img0 = dataset[kk, :, :, :].numpy()
        plt.imshow(img0/255)
        plt.title(f'label={target[kk]}')
        plt.show()



if __name__ == '__main__':
    main()