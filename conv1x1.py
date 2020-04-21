"""
1x1 conv filter illustration
"""

import torch
import torch.nn.functional as F
import pickle
import numpy as np

import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def main():
    file_path = './data/cifar10_singlebatch/data_batch_1'
    dataset = unpickle(file_path)
    data = dataset['data']

    img = data[0,:]
    R = img[0:1024].reshape(32,32)/255.0
    G = img[1024:2048].reshape(32,32)/255.0
    B = img[2048:].reshape(32,32)/255.0

    X = np.dstack((R,G,B))

    
    X_t = torch.FloatTensor(X)
    
    weights = 0.5*torch.ones(1,3,1,1)
    weights_2 = 0.5*torch.ones(1,3,5,5)

    Y = F.conv2d(X_t.view(1,3,32,32), weights)
    Y2 = F.conv2d(X_t.view(1,3,32,32), weights_2)

    print(Y.size())
    plt.figure(figsize=(16,6))
    plt.subplot(141)
    plt.imshow(X.reshape(32,32,3))
    plt.xticks([])
    plt.yticks([])
    plt.title('Input: Frog')

    plt.subplot(142)
    plt.imshow(weights_2[0].reshape(5,5,3))
    plt.xticks([])
    plt.yticks([])
    plt.title('weights')    

    plt.subplot(143)
    plt.imshow(Y.numpy().reshape(32,32))
    plt.xticks([])
    plt.yticks([])
    plt.title('Conv1x1 Output: Frog')

    plt.subplot(144)
    plt.imshow(Y2.numpy().reshape(28,28))
    plt.xticks([])
    plt.yticks([])
    plt.title('Conv5x5 Output: Frog')
    plt.savefig('conv1x1_img.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)
    

if __name__ == '__main__':
    main()
