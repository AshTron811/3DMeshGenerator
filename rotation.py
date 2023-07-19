import torch as torch
from torch.utils.data import DataLoader
import numpy as np


def rotate(s, theta=0, axis='x'):
    """
    Rotation of a point cloud by theta degrees along axis

    Args:
        s (array): A numpy array of shape (N, 3)
        theta (int): Angle [0,360] of rotation
        axis (str): Either 'x', 'y', or 'z'

    Returns:
        array: Input array s with rotation performed
    """
    theta = np.radians(theta)  # degree -> radians
    r = 0
    if axis.lower() == 'x':
        r = [s[0],
             s[1] * np.cos(theta) - s[2] * np.sin(theta),
             s[1] * np.sin(theta) + s[2] * np.cos(theta)]
    elif axis.lower() == 'y':
        r = [s[0] * np.cos(theta) + s[2] * np.sin(theta),
             s[1],
             -s[0] * np.sin(theta) + s[2] * np.cos(theta)]
    elif axis.lower() == 'z':
        r = [s[0] * np.cos(theta) - s[1] * np.sin(theta),
             s[0] * np.sin(theta) + s[1] * np.cos(theta),
             s[2]]
    else:
        print("Error! Invalid axis rotation")
    return r


BATCH_SIZE = 25

# Replace this with where you saved your .npy files in the last section
examples = []
point_cloud_collections = []

point_cloud = np.load('pcloud.npy')

# Apply random rotation

theta = np.random.randint(low=0, high=360)

pc1 = point_cloud
pc2 = rotate(point_cloud, theta)

# Move axis so that channels are first like pytorch wants it by default

point_cloud = np.moveaxis(pc1, -1, 0)
point_cloud2 = np.moveaxis(pc2, -1, 0)
if point_cloud.shape != point_cloud2.shape:
    print("Rotation was not succesful")

point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
point_cloud2 = torch.from_numpy(point_cloud2.astype(np.float32))

if point_cloud.shape != point_cloud2.shape:
    print("Tensor conversion was not successful")
point_cloud_collections.append(point_cloud)
point_cloud_collections.append(point_cloud2)

train_data = point_cloud_collections

print("done loading")

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

for step, pt in enumerate(train_loader):
    print(pt.shape)
    break
print("Total training examples: " + str(len(train_data)))