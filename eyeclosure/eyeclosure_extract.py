"""

"""
import numpy as np
import torch
from six.moves import cPickle as pickle

pickle_files = ['./open_eyes.pickle', './closed_eyes.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            print("here")
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save  # hint to help gc free up memory
    i += 1

train_dataset_tensor = torch.Tensor(train_dataset)
train_label_tensor = torch.Tensor(train_labels).squeeze(1)

test_dataset_tensor = torch.Tensor(test_dataset)
test_label_tensor = torch.Tensor(test_labels).squeeze(1)

print('Training set', train_dataset_tensor.size(), train_label_tensor.size())
print('Test set', test_dataset_tensor.size(), test_label_tensor.size())

torch.save(train_dataset_tensor, './eyeclosure_train_data.pt')
torch.save(train_label_tensor, './eyeclosure_train_label.pt')

torch.save(test_dataset_tensor, './eyeclosure_test_data.pt')
torch.save(test_label_tensor, './eyeclosure_test_label.pt')