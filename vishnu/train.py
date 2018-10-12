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

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

current_directory = os.getcwd()  # current working directory
training_ratio = 0.8

if __name__ == '__main__':
    polygons_label_path = "/home/datasets/full_dataset_labels/train_extra"
    images_path = "/home/datasets/full_dataset/train_extra"

    compl_poly_path = os.path.join(polygons_label_path, "*", "*_polygons.json")

    poly_folders = glob(compl_poly_path)

    poly_folders = np.array(poly_folders)

    image_label_list = []

    for file in poly_folders:
        with open(file, "r") as f:
            frame_info = json.load(f)
            length = len(frame_info['objects'])
            file_path = file
            image_name = file_path.split("/")[-1][:-23]
            for i in range(length):
                label = frame_info['objects'][i]['label']
                if label == "ego vehicle":
                    break
                polygon = np.array(frame_info['objects'][i]['polygon'], dtype=np.float32)
                left_top = np.min(polygon, axis=0)
                right_bottom = np.max(polygon, axis=0)
                ltrb = np.concatenate((left_top, right_bottom))
                image_label_list.append(
                    {'image_name': image_name, 'file_path': file_path, 'label': label, 'bbox': ltrb})

    image_ll_len = len(image_label_list)

    # get images list

    compl_img_path = os.path.join(images_path, "*", "*")

    images = glob(compl_img_path)

    images = np.array(images)

    print("creating image data list")
    print(len(images))

    # lsit = []
    #
    # for i in range(image_ll_len):
    #     lsit.append(image_label_list[i]['label'])
    #
    # print(np.asarray(set(lsit)))

    train_valid_datlist = []
    for i in range(0, len(images)):
        img_folder = images[i].split('/')[-2]
        img_name = images[i].split('/')[-1]
        img_iden = img_name[:-16]
        image_path = os.path.join(images_path, img_folder, img_name)
        print(image_path)
        b_boxes = []
        labels = []
        for i in range(image_ll_len):
            if image_label_list[i]["image_name"] == img_iden:
                bbox = image_label_list[i]['bbox']
                b_boxes.append(bbox)
                if image_label_list[i]['label'] in ('car', 'cargroup'):
                    label = 1
                elif image_label_list[i]['label'] in ('person', 'persongroup'):
                    label = 2
                elif image_label_list[i]['label'] == 'traffic sign':
                    label = 3
                else:
                    label = 0
                labels.append(label)
        train_valid_datlist.append({'image_path': image_path, 'labels': labels, 'bboxes': b_boxes})

    random.shuffle(train_valid_datlist)
    total_training_validation_items = len(train_valid_datlist)

    # Training dataset.
    n_train_sets = training_ratio * total_training_validation_items
    train_set_list = train_valid_datlist[: int(n_train_sets)]

    # Validation dataset.
    n_valid_sets = (1 - training_ratio) * total_training_validation_items
    valid_set_list = train_valid_datlist[int(n_train_sets): int(n_train_sets + n_valid_sets)]

    train_dataset = cityscape_dataset.CityScapeDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    print('Total training items', len(train_dataset), ', Total training batches per epoch:', len(train_data_loader))

    valid_dataset = cityscape_dataset.CityScapeDataset(valid_set_list)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=6)
    print('Total training items', len(valid_dataset), ', Total training batches per epoch:', len(valid_data_loader))

    # train_batch_idx, (train_input, train_label) = next(enumerate(train_data_loader))
    net = ssd_net.SSD(num_classes=4)

    criterion = bbox_loss.MultiboxLoss(bbox_pre_var=[0.1, 0.2])

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    print("start train")

    itr = 0
    max_epochs = 3
    train_losses = []
    valid_losses = []

    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (images, loc_targets, conf_targets) in enumerate(train_data_loader):
            itr += 1
            net.train()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            images = Variable(images.cuda())  # Use Variable(*) to allow gradient flow
            loc_targets = Variable(loc_targets.cuda())
            conf_targets = Variable(conf_targets.cuda()).long()
            conf_preds, loc_preds = net.forward(images)  # Forward once

            # Compute loss
            # forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_loc):
            loss = criterion(conf_preds, loc_preds, conf_targets, loc_targets)

            # Do the backward and compute gradients
            loss.backward()

            # Update the parameters with SGD
            optimizer.step()

            train_losses.append((itr, loss.item()))

            if train_batch_idx % 50 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

                # Run the validation every 200 iteration:
            if train_batch_idx % 50 == 0:
                net.eval()  # [Important!] set the network in evaluation model
                valid_loss_set = []  # collect the validation losses
                valid_itr = 0

                # Do validation
                for valid_batch_idx, (valid_img, valid_locs, valid_labels) in enumerate(valid_data_loader):
                    net.eval()
                    valid_img = Variable(valid_img.cuda())  # use Variable(*) to allow gradient flow
                    v_pred_conf, v_pred_locs = net.forward(valid_img)  # forward once

                    # Compute loss
                    valid_locs = Variable(valid_locs.cuda())
                    valid_labels = Variable(valid_labels.cuda())
                    valid_loss = criterion(v_pred_conf, v_pred_locs, valid_labels, valid_locs)
                    valid_loss_set.append(valid_loss.item())

                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, float(avg_valid_loss)))
                valid_losses.append((itr, avg_valid_loss))

    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)
    train_losses[0] = train_losses[1]
    valid_losses[0] = valid_losses[1]

    net_state = net.state_dict()  # serialize trained model
    torch.save(net_state, '/home/vramiyas/SSDnet_1.pth')  # save to disk



    print(valid_losses)



    plt.plot(train_losses[2:, 0],  # Iteration
             train_losses[2:, 1])  # Loss value
    plt.plot(valid_losses[2:, 0],  # Iteration
             valid_losses[2:, 1])  # Loss value
    plt.show()
