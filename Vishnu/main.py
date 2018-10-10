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
#import matplotlib.pyplot as plt
#import matplotlib.patches as pat
import torch
import random

# Set default tenosr type, 'torch.cuda.FloatTensor' is the GPU based FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')

current_directory = os.getcwd()  # current working directory
training_ratio = 0.8

if __name__ == '__main__':
    polygons_label_path = "cityscapes_samples_labels"
    images_path = "cityscapes_samples"

    compl_poly_path = os.path.join(current_directory, polygons_label_path, "*", "*_polygons.json")

    poly_folders = glob(compl_poly_path)

    poly_folders = np.array(poly_folders)

    print(poly_folders.shape)

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
                ltrb = np.concatenate((left_top, right_bottom)).reshape(1, -1)
                image_label_list.append(
                    {'image_name': image_name, 'file_path': file_path, 'label': label, 'bbox': ltrb})

    print(image_label_list)

    image_ll_len = len(image_label_list)

    # get images list

    compl_img_path = os.path.join(current_directory, images_path, "*", "*")

    images = glob(compl_img_path)

    images = np.array(images)

    print(len(images))

    # lsit = []
    #
    # for i in range(image_ll_len):
    #     lsit.append(image_label_list[i]['label'])
    #
    # print(np.asarray(set(lsit)))

    train_valid_datlist = []
    for i in range(0,len(images)):
        print(images[i])
        print(images[i].split("/"))
        img_folder = images[i].split('/')[-2]
        img_name = images[i].split('/')[-1]
        print(img_folder)
        print(img_name[:-16])
        img_iden = img_name[:-16]
        image_path = os.path.join(current_directory, images_path, img_folder, img_name)
        b_boxes = []
        labels = []
        for i in range(image_ll_len):
            if image_label_list[i]["image_name"] == img_iden:
                bbox = image_label_list[i]['bbox']
                b_boxes.append(bbox)
                if image_label_list[i]['label'] in  ('car','cargroup'):
                    label = 1
                elif image_label_list[i]['label'] in ('person','persongroup') :
                    label = 2
                elif image_label_list[i]['label'] == 'traffic sign':
                    label = 3
                else:
                    label = 0
                labels.append(label)
        train_valid_datlist.append({'image_path': image_path, 'labels': labels, 'bboxes': b_boxes})

    random.shuffle(train_valid_datlist)
    total_training_validation_items = len(train_valid_datlist)
    print
    print(len(train_valid_datlist))


    # Training dataset.
    n_train_sets = training_ratio * total_training_validation_items
    train_set_list = train_valid_datlist[: int(n_train_sets)]

    # Validation dataset.
    n_valid_sets = (1 - training_ratio) * total_training_validation_items
    valid_set_list = train_valid_datlist[int(n_train_sets): int(n_train_sets + n_valid_sets)]

    train_dataset = cityscape_dataset.CityScapeDataset(train_set_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print('Total training items', len(train_dataset), ', Total training batches per epoch:', len(train_data_loader))

    train_batch_idx, (train_input, train_label) =  next(enumerate(train_data_loader))

