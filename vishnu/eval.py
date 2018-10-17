import time
import ssd_net
import mobilenet
import bbox_loss
import cityscape_dataset
import bbox_helper
import module_util
import os
from glob import glob
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import torch
import random
import torch.optim
from torch.autograd import Variable
import pickle

torch.set_default_tensor_type('torch.cuda.FloatTensor')
current_directory = os.getcwd()  # current working directory

if __name__ == '__main__':
    net = ssd_net.SSD(num_classes=5)

    model_path = os.path.join(current_directory, 'trained_models')

    net_state = torch.load(model_path)

    net.load_state_dict(net_state)

    with open('saved_list', 'rb') as fp:
        test_datlist = pickle.load(fp)

    print(len(test_datlist))

    random.shuffle(test_datlist)

    test_dataset = cityscape_dataset.CityScapeDataset(test_datlist)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Total training items', len(test_dataset), ', Total training batches per epoch:', len(test_data_loader))
    print("batch_size : ", 1)

    test_idx, (test_img, test_bboz, test_labels, prior_bboxes) = next(enumerate(test_data_loader))

    net.cuda()

    net.eval()


