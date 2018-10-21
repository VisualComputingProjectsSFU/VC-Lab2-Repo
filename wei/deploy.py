import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ssd_net
import cityscape_dataset
import bbox_helper as helper

# Configurations
threshold = 0.95

# Prepare the network.
dataset = cityscape_dataset.CityScapeDataset([], 0)
net = ssd_net.SsdNet(len(dataset.classes), dataset.num_prior_bbox)
net_state = torch.load('ssd_net.pth')
net.load_state_dict(net_state)
net.eval()
net.cpu()

# Prepare the input.
original_input = Image.open('sample.png')
sample = original_input
sample = np.array(sample)
sample = (sample / 255) * 2 - 1
sample = sample.reshape((sample.shape[2], sample.shape[0], sample.shape[1]))
sample = torch.Tensor(sample)
sample = sample.unsqueeze(0).cpu()

# Detect the object.
confidences, locations = net.forward(sample)
confidences = confidences.detach().numpy()
max_vector = np.amax(confidences[0, ...], axis=1)
print('Number of bounding boxes generated: ' + str((max_vector > threshold).sum()))

# Plot the result.
locations = torch.mul(locations, 300)
matched_rect = None

fig, ax = plt.subplots(1)
ax.imshow(np.array(original_input))
for index in range(0, max_vector.shape[0]):
    if max_vector[index] > threshold:
        corner = helper.center2corner(locations[0, index])
        torch.clamp(corner, 0, 300)
        x = corner[0]
        y = corner[1]
        matched_rect = patches.Rectangle(
            (x, y),
            locations[0, index, 2],
            locations[0, index, 3],
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(matched_rect)

fig.canvas.set_window_title('Sample Detection')
plt.title('Sample Detection')
plt.xlim(0, 300)
plt.ylim(300, 0)
if matched_rect is not None:
    plt.legend([matched_rect],
               ['Detected Objects'],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()

plt.show()
