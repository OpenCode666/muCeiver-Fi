"""
load data
"""
import sys
import torch
import numpy as np
import h5py
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset


def load_data(in_dir, out_dir, batch_size,shuffle):

    input_data = np.load(in_dir)
    if 'test_data' in input_data.keys():
        input_data = input_data['test_data']
    elif 'train_data' in input_data.keys():
        input_data = input_data['train_data']
    else:
        print('Wrong Filename...')
        sys.exit()
    # input_data = np.transpose(input_data, (2, 3, 1, 0))   # [128, 60, channel, samples]

    output_data = np.load(out_dir)
    if 'test_out' in output_data.keys():
        output_data = output_data['test_out']
    elif 'train_out' in output_data.keys():
        output_data = output_data['train_out']
    else:
        print('Wrong Filename...')
        sys.exit()
    # output_data = np.transpose(output_data, (1, 2, 0))    # [30,50,samples]

    x_data = input_data.astype(np.float32)
    y_data = np.expand_dims(output_data, axis=1).astype(np.float32)
    
    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats

