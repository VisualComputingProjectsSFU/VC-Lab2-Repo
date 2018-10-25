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
import matplotlib.patches as patches

torch.set_default_tensor_type('torch.cuda.FloatTensor')
current_directory = os.getcwd()  # current working directory

if __name__ == '__main__':
    # polygons_label_path = "/home/datasets/full_dataset_labels/train_extra"
    # images_path = "/home/datasets/full_dataset/train_extra"
    polygons_label_path = os.path.join(current_directory, "polygons")
    images_path = os.path.join(current_directory, "images")

    torch.multiprocessing.set_start_method("spawn")

    # compl_poly_path = os.path.join(polygons_label_path, "*_polygons.json")
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
    # compl_img_path = os.path.join(images_path, "*")
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
    #     image_path = os.path.join(images_path, img_name)
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
    #
    # outfile = os.path.join(current_directory, 'test_list')
    #
    # with open(outfile, 'wb') as fp:
    #     pickle.dump(train_valid_datlist, fp)
    # #
    # exit()

    # with open(outfile, 'rb') as fp:
    #     train_valid_datlist = pickle.load(fp)

    net = ssd_net.SSD(num_classes=6)

    model_path = os.path.join(current_directory, 'SSDnet_crop1.pth')

    net_state = torch.load(model_path)

    net.load_state_dict(net_state)

    with open('test_list', 'rb') as fp:
        test_datlist = pickle.load(fp)

    print(len(test_datlist))

    random.shuffle(test_datlist)

    test_dataset = cityscape_dataset.CityScapeDataset(test_datlist)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Total training items', len(test_dataset), ', Total training batches per epoch:', len(test_data_loader))
    print("batch_size : ", 1)

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

    test_idx, (test_img, test_bbox, test_labels) = next(enumerate(test_data_loader))

    net.cuda()

    net.eval()

    images = Variable(test_img.cuda())  # Use Variable(*) to allow gradient flow\n",
    loc_targets = Variable(test_bbox.cuda())
    conf_targets = Variable(test_labels.cuda()).long()
    conf_preds, loc_preds = net.forward(images)  # Forward once\n",
    print(conf_preds.shape)
    print(loc_preds.shape)

    bbox = bbox_helper.loc2bbox(loc_preds, prior_bboxes)
    bbox = bbox[0].detach()
    bbox_corner = bbox_helper.center2corner(bbox)
    print(bbox_corner)
    print(conf_preds)
    print(bbox_corner.shape)
    bbox_corner = bbox_corner
    conf_preds = conf_preds[0].detach()

    # idx = conf_preds[:, 2] > 0.6
    # bbox_corner = bbox_corner[idx]
    # bbox = bbox[idx]
    # print(bbox_corner)

    bbox_nms = bbox_helper.nms_bbox(bbox_corner,conf_preds)
    print(bbox_nms[0])
    bbox_nms = torch.Tensor(bbox_nms[0])
    print(bbox_nms.shape)
    bbox_nms_cen = bbox_helper.corner2center(bbox_nms)

    test_img = test_img.detach()
    channels = test_img.shape[1]
    h, w = test_img.shape[2], test_img.shape[3]

    img_r = test_img.reshape(h, w, channels)
    img_n = (img_r + 1) / 2

    fig, ax = plt.subplots(1)

    ax.imshow(img_n)

    for index in range(0, bbox_nms.shape[0]):
        corner = bbox_nms[index]
        corner = torch.mul(corner, 300)
        x = corner[0]
        y = corner[1]
        raw_matched_rect = patches.Rectangle(
            (x, y),
            bbox_nms_cen[index, 2] * 300,
            bbox_nms_cen[index, 3] * 300,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(raw_matched_rect)

    plt.show()

    # bbox_nms = bbox_helper.nms_bbox(bbox_corner,conf_preds)
    # print(bbox_nms)
