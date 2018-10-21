import os
import datetime
import json
import numpy as np
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import ssd_net
import cityscape_dataset
import bbox_loss

learning_rate = 0.01
num_worker = 6

# Detect data location.
data_path = os.path.join('cityscapes_samples')
label_path = os.path.join('cityscapes_samples_labels')

if not os.path.isdir(data_path):
    data_path = os.path.join('/home/datasets/full_dataset/train_extra')
    label_path = os.path.join('/home/datasets/full_dataset_labels/train_extra')

# Default configurations.
np.set_printoptions(suppress=True)


def collate_fn(batch):
    print('Output of Batch')
    print(len(batch))
    print('Batch Len')
    for i in range(0, len(batch)):
        print('i')
        print(len(batch[i]))
        for j in range(0, len(batch[i])):
            print('j')
            print(len(batch[j]))
    print(batch)
    f = open("out.txt", "w+")
    f.write(str(batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # Read data.
    dir_list = os.listdir(data_path)
    data_list = {}

    counter = 1
    num_total_file = sum([len(files) for r, d, files in os.walk(data_path)])

    print('Reading Dataset Files')
    for directory in dir_list:
        data_dir = os.path.join(data_path, directory)
        label_dir = os.path.join(label_path, directory)
        file_list = os.listdir(data_dir)

        # Read label.
        for i_data in range(0, len(file_list)):
            label = label_dir
            label = os.path.join(label, file_list[i_data][:-15] + 'gtCoarse_polygons.json')
            with open(label) as file:
                data_list[counter - 1] = json.load(file)
                file = os.path.join(data_dir, file_list[i_data])
                data_list[counter - 1]['file'] = file

                # Print the loading.
                print(str(counter) + '/' + str(num_total_file), end='', flush=True)
                print('\r', end='', flush=True)
                counter += 1

    # Create dataset.
    print('Creating Dataset & Dataloader')
    dataset = cityscape_dataset.CityScapeDataset(data_list, max(1, num_worker))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=num_worker, collate_fn=collate_fn)

    # Losses collection, used for monitoring over-fit.
    train_losses = []

    net = ssd_net.SsdNet(len(dataset.classes), dataset.num_prior_bbox)
    max_epochs = 10
    iteration = 0
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = bbox_loss.MultiboxLoss()

    for epoch_idx in range(0, max_epochs):
        start_time = datetime.datetime.now()
        print('===============================================================================')
        print('= Epoch {:2d} start at: '.format(epoch_idx) + str(start_time) + '                               =')
        print('===============================================================================')

        for train_batch_idx, (inp, confidence_oracles, location_oracles) in enumerate(data_loader):
            print('INSIDE1')
            iteration += 1

            net.train()
            net.cuda()

            print('INSIDE2')

            # Zero the parameter gradients.
            optimizer.zero_grad()

            print('INSIDE3')

            # Forward.
            inp = Variable(inp)
            confidence_predictions, location_predictions = net.forward(inp)

            print('INSIDE4')

            # Compute loss.
            confidence_oracles = Variable(confidence_oracles)
            location_oracles = Variable(location_oracles)
            loss = criterion(confidence_predictions, location_predictions, confidence_oracles, location_oracles)

            print('INSIDE5')

            # Skip when there is no positive detected.
            if type(loss) == int:
                print('SKIP')
                continue

            # Do the backward and compute gradients.
            loss.backward()

            print('INSIDE6')

            # Update the parameters with SGD.
            optimizer.step()

            print('INSIDE7')

            # Add the tuple of（iteration, loss) into `train_losses` list.
            train_losses.append((iteration, loss.item()))

            if train_batch_idx % 200 == 0:
                print('Epoch: {:d} Iteration: {:d} Loss: {:f}'.format(epoch_idx, iteration, loss.item()))

        # Save results for every epoch.
        net_state = net.state_dict()
        torch.save(net_state, 'ssd_net.pth')

    train_losses = np.asarray(train_losses)
    if len(train_losses) > 0:
        train_losses[0] = train_losses[1]

    plt.plot(train_losses[:, 0],      # Iteration.
             train_losses[:, 1])      # Loss value.

    plt.show()
