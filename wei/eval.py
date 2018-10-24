import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ssd_net
import cityscape_dataset
import bbox_helper as helper

np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument('filename', default='sample.png')
parser.add_argument('threshold', default='0.4')
arguments = parser.parse_args()

# Configurations
threshold = float(arguments.threshold)
filename = arguments.filename

# Prepare the network.
dataset = cityscape_dataset.CityScapeDataset([], 0)
net = ssd_net.SsdNet(len(dataset.classes), dataset.num_prior_bbox)
net_state = torch.load('ssd_net.pth')
net.load_state_dict(net_state)
net.eval()
net.cpu()

# Prepare the input.
original_input = Image.open(filename)
sample = original_input
sample = np.array(sample)
sample = np.subtract(sample, [127, 127, 127])
sample = np.divide(sample, 128)
sample = sample.reshape((sample.shape[2], sample.shape[0], sample.shape[1]))
sample = torch.Tensor(sample)
sample = sample.unsqueeze(0).cpu()

# Detect the object.
raw_confidences, raw_locations = net.forward(sample)
fine_confidences, fine_locations = helper.nms_bbox(
    raw_confidences[0, :, 1:].detach(), raw_locations[0].detach(), prob_threshold=threshold)
raw_confidences = raw_confidences.detach().numpy()
max_vector = np.amax(raw_confidences[0, :, 1:], axis=-1)
print('               File Name: ' + filename)
print('    Confidence Threshold: ' + str(threshold))
print('Raw Bounding Boxes Found: ' + str((max_vector > threshold).sum()))

# Plot the result.
raw_matched_rect = None
fine_matched_rect = None

fig, ax = plt.subplots(1)
ax.imshow(np.array(original_input))

# Display raw predictions.
for index in range(0, max_vector.shape[0]):
    if max_vector[index] > threshold:
        corner = helper.center2corner(raw_locations[0, index])
        corner = torch.mul(corner, 300)
        corner = torch.clamp(corner, 0, 300)
        x = corner[0]
        y = corner[1]
        raw_matched_rect = patches.Rectangle(
            (x, y),
            raw_locations[0, index, 2] * 300,
            raw_locations[0, index, 3] * 300,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(raw_matched_rect)

# Display fine predictions.
for location in fine_locations:
    corner = helper.center2corner(location)
    corner = torch.mul(corner, 300)
    corner = torch.clamp(corner, 0, 300)
    x = corner[0]
    y = corner[1]
    fine_matched_rect = patches.Rectangle(
        (x, y),
        location[2] * 300,
        location[3] * 300,
        linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(fine_matched_rect)

fig.canvas.set_window_title('Sample Detection')
plt.title('Sample Detection')
plt.xlim(0, 300)
plt.ylim(300, 0)
if raw_matched_rect is not None:
    plt.legend([raw_matched_rect, fine_matched_rect],
               ['Raw Detected Objects', 'Refined Detected Objects'],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()

plt.show()
