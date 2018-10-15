import os
import json
import random
import numpy as np
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import ssd_net
import cityscape_dataset
import bbox_loss

data_path = os.path.join('cityscapes_samples', 'bad-honnef')
label_path = os.path.join('cityscapes_samples_labels', 'bad-honnef')
learning_rate = 0.0005
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    # Read data.
    file_list = os.listdir(data_path)
    data_list = {}

    # Read label.
    for i_data in range(0, len(file_list)):
        label = label_path
        label = os.path.join(label, file_list[i_data][:-15] + 'gtCoarse_polygons.json')
        with open(label) as file:
            data_list[i_data] = json.load(file)
            data_list[i_data]['file'] = file_list[i_data]

    random.shuffle(data_list)

    # Create dataset.
    dataset = cityscape_dataset.CityScapeDataset(data_list)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)

    dataset.preview(15)

    # Losses collection, used for monitoring over-fit.
    train_losses = []

    net = ssd_net.SsdNet(len(dataset.classes), dataset.num_prior_bbox)
    max_epochs = 10
    itr = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = bbox_loss.MultiboxLoss()

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_input, train_oracle) in enumerate(data_loader):
            itr += 1
            net.train()
            net.cuda()

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward.
            train_input = Variable(train_input.cuda())  # Use Variable(*) to allow gradient flow.
            train_out = net.forward(train_input)  # Forward once.

            # Compute loss.
            train_oracle = Variable(train_oracle.cuda())
            loss = criterion(train_out, train_oracle)

            # Do the backward and compute gradients.
            loss.backward()

            # Update the parameters with SGD.
            optimizer.step()

            # Add the tuple of（iteration, loss) into `train_losses` list.
            train_losses.append((itr, loss.item()))

            if train_batch_idx % 200 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

    net_state = net.state_dict()
    torch.save(net_state, 'lfw_net.pth')

    train_losses = np.asarray(train_losses)
    train_losses[0] = train_losses[1]

    plt.plot(train_losses[:, 0],      # Iteration.
             train_losses[:, 1])      # Loss value.

    plt.show()
