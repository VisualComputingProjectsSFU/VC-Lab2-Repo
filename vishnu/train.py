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
import torch.nn.functional as F

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

current_directory = os.getcwd()  # current working directory
training_ratio = 0.8
validation_ratio = 0.5

if __name__ == '__main__':

    polygons_label_path = "/home/datasets/full_dataset_labels/train_extra"
    images_path = "/home/datasets/full_dataset/train_extra"
    # polygons_label_path = os.path.join(current_directory, "cityscapes_samples_labels")
    # images_path = os.path.join(current_directory, "cityscapes_samples")

    torch.multiprocessing.set_start_method("spawn")

    # compl_poly_path = os.path.join(polygons_label_path, "*", "*_polygons.json")
    #
    # poly_folders = glob(compl_poly_path)
    #
    # poly_folders = np.array(poly_folders)
    #
    # image_label_list = []
    #
    # for file in poly_folders:
    #     with open(file, "r") as f:
    #         frame_info = json.load(f)
    #         length = len(frame_info['objects'])
    #         file_path = file
    #         image_name = file_path.split("/")[-1][:-23]
    #         for i in range(length):
    #             label = frame_info['objects'][i]['label']
    #             if label == "ego vehicle":
    #                 break
    #             polygon = np.array(frame_info['objects'][i]['polygon'], dtype=np.float32)
    #             left_top = np.min(polygon, axis=0)
    #             right_bottom = np.max(polygon, axis=0)
    #             ltrb = np.concatenate((left_top, right_bottom))
    #             if ltrb.shape[0] != 4:
    #                 print(file_path)
    #             image_label_list.append(
    #                 {'image_name': image_name, 'file_path': file_path, 'label': label, 'bbox': ltrb})
    #
    # image_ll_len = len(image_label_list)
    #
    # # get images list
    #
    # compl_img_path = os.path.join(images_path, "*", "*")
    #
    # images = glob(compl_img_path)
    #
    # images = np.array(images)
    #
    # print("creating image data list")
    # print(len(images))
    #
    # # lsit = []
    # #
    # # for i in range(image_ll_len):
    # #     lsit.append(image_label_list[i]['label'])
    # #
    # # print(np.asarray(set(lsit)))
    #
    #
    # curr_folder = 'a'
    # train_valid_datlist = []
    # for i in range(0, len(images)):
    #     img_folder = images[i].split('/')[-2]
    #     img_name = images[i].split('/')[-1]
    #     img_iden = img_name[:-16]
    #     image_path = os.path.join(images_path, img_folder, img_name)
    #     if img_folder != curr_folder:
    #         print(img_folder)
    #         print(image_path)
    #     curr_folder = img_folder
    #     b_boxes = []
    #     labels = []
    #     cnt = 0
    #     for i in range(image_ll_len):
    #         if image_label_list[i]["image_name"] == img_iden:
    #             if image_label_list[i]['label'] == 'car':
    #                 label = 1
    #                 cnt += 1
    #                 bbox = image_label_list[i]['bbox']
    #                 b_boxes.append(bbox)
    #                 labels.append(label)
    #             elif image_label_list[i]['label'] == 'cargroup':
    #                 cnt += 1
    #                 label = 2
    #                 bbox = image_label_list[i]['bbox']
    #                 b_boxes.append(bbox)
    #                 labels.append(label)
    #             elif image_label_list[i]['label'] == 'person':
    #                 cnt += 1
    #                 label = 3
    #                 bbox = image_label_list[i]['bbox']
    #                 b_boxes.append(bbox)
    #                 labels.append(label)
    #             elif image_label_list[i]['label'] == 'persongroup':
    #                 cnt += 1
    #                 label = 4
    #                 bbox = image_label_list[i]['bbox']
    #                 b_boxes.append(bbox)
    #                 labels.append(label)
    #             elif image_label_list[i]['label'] == 'traffic sign':
    #                 label = 5
    #                 bbox = image_label_list[i]['bbox']
    #                 b_boxes.append(bbox)
    #                 labels.append(label)
    #     if cnt == 0:
    #         continue
    #     train_valid_datlist.append({'image_path': image_path, 'labels': labels, 'bboxes': b_boxes})

    outfile = os.path.join(current_directory, 'saved_list2')

    # with open(outfile, 'wb') as fp:
    #     pickle.dump(train_valid_datlist, fp)
    #
    # exit()

    with open(outfile, 'rb') as fp:
        train_valid_datlist = pickle.load(fp)

    print(len(train_valid_datlist))

    train_valid_datlist = train_valid_datlist

    random.shuffle(train_valid_datlist)
    total_training_validation_items = len(train_valid_datlist)
    batch_size = 16

    # Training dataset.
    n_train_sets = training_ratio * total_training_validation_items
    train_set_list = train_valid_datlist[: int(n_train_sets)]

    # Validation dataset.
    n_valid_sets = (total_training_validation_items - n_train_sets ) * validation_ratio
    valid_set_list = train_valid_datlist[int(n_train_sets):int(n_train_sets+n_valid_sets)]

    # Test dataset.
    n_test_sets = (total_training_validation_items - n_train_sets ) * validation_ratio
    test_set_list = train_valid_datlist[
                    int(n_train_sets + n_valid_sets): int(n_train_sets + n_valid_sets + n_test_sets)]

    with open('train_list', 'wb') as fp:
        pickle.dump(train_set_list, fp)

    with open('valid_list', 'wb') as fp:
        pickle.dump(valid_set_list, fp)

    with open('test_list', 'wb') as fp:
        pickle.dump(test_set_list, fp)




    train_dataset = cityscape_dataset.CityScapeDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Total training items', len(train_dataset), ', Total training batches per epoch:', len(train_data_loader))
    print("batch_size : ", batch_size)

    valid_dataset = cityscape_dataset.CityScapeDataset(valid_set_list)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Total validation items', len(valid_dataset), ', Total validation batches per epoch:', len(valid_data_loader))

    # train_batch_idx, (train_input, train_label) = next(enumerate(train_data_loader))
    net = ssd_net.SSD(num_classes=6)

    net = net.cuda()

    criterion = bbox_loss.MultiboxLoss()

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    print("start train")

    itr = 0
    max_epochs = 10
    train_losses = []
    valid_losses = []

    start_time = time.time()

    model_path = os.path.join(current_directory, 'SSDnet_crop1.pth')

    net_state = torch.load(model_path)

    net.load_state_dict(net_state)

    curr_epoch = 0

    prior_layer_cfg = [{'layer_name': 'Conv11', 'feature_dim_hw': (19, 19), 'bbox_size': (60, 60),
                        'aspect_ratio': [2, 3, 4]},
                       {'layer_name': 'Conv13', 'feature_dim_hw': (10, 10), 'bbox_size': (102, 102),
                        'aspect_ratio': [2, 3, 4]},
                       {'layer_name': 'Conv14_2', 'feature_dim_hw': (5, 5), 'bbox_size': (144, 144),
                        'aspect_ratio': [2, 3, 4]},
                       {'layer_name': 'Conv15_2', 'feature_dim_hw': (3, 3), 'bbox_size': (186, 186),
                        'aspect_ratio': [2]},
                       {'layer_name': 'Conv16_2', 'feature_dim_hw': (2, 2), 'bbox_size': (228, 228),
                        'aspect_ratio': [2]},
                       {'layer_name': 'Conv16_2', 'feature_dim_hw': (1, 1), 'bbox_size': (270, 270),
                        'aspect_ratio': [2]}
                       ]
    prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg)
    prior_bboxes = prior_bboxes.unsqueeze(0)

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


            if loss == Variable(torch.Tensor([0])):
                continue

            loss.backward()

            # Update the parameters with SGD
            optimizer.step()

            train_losses.append((itr, loss.item()))

            if train_batch_idx % 50 == 0:
                print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))
                print('elasped time:', time.time() - start_time)
                print('Training  Loss', loss)

                # Run the validation every 200 iteration:
            if train_batch_idx % 200 == 0:
                net.eval()  # [Important!] set the network in evaluation model
                valid_loss_set = []  # collect the validation losses
                valid_itr = 0

                # Do validation
                for valid_batch_idx, (valid_img, valid_locs, valid_labels) in enumerate(valid_data_loader):
                    valid_img = Variable(valid_img.cuda())  # use Variable(*) to allow gradient flow
                    v_pred_conf, v_pred_locs = net.forward(valid_img)  # forward once

                    for i in range(batch_size):
                        index = valid_labels[i] > 0
                        if index.data.sum() > 0:
                            print(valid_labels[i,index])
                            out = F.softmax(v_pred_conf[i]).detach()
                            print('postives : ', out[index])
                            locs = v_pred_locs[i]
                            print('locs', locs[index])
                            locs = locs.unsqueeze(0)
                            bbox = bbox_helper.loc2bbox(locs,prior_bboxes)
                            print('bboxes',bbox[0,index])
                            break




                    # Compute loss
                    valid_locs = Variable(valid_locs.cuda())
                    valid_labels = Variable(valid_labels.cuda()).long()
                    valid_loss = criterion(v_pred_conf, v_pred_locs, valid_labels, valid_locs)
                    print('Validation Loss', valid_loss)
                    valid_loss_set.append(valid_loss.item())

                    valid_itr += 1
                    if valid_itr > 5:
                        break

                # Compute the avg. validation loss
                avg_valid_loss = np.mean(np.asarray(valid_loss_set))
                print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, float(avg_valid_loss)))
                valid_losses.append((itr, avg_valid_loss))

            if epoch_idx != curr_epoch:
                net_state = net.state_dict()  # serialize trained model
                torch.save(net_state, '/home/vramiyas/SSDnet_crop1.pth')  # save to disk

            curr_epoch = epoch_idx
    net_state = net.state_dict()  # serialize trained model
    torch.save(net_state, '/home/vramiyas/SSDnet_crop1.pth')  # save to disk
