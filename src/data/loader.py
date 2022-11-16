#Load libraries
import os
import torchvision
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import PILToTensor, Compose
from torch.utils.data import Dataset
from os import listdir
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self,  img_dir, transform=None, target_transform=None):
        self.img_labels = [[x, 0] for x in listdir(img_dir)]
        self.img_labels.pop(0) #delete ['.gitkeep', 0] 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels[idx][0]))
        image = read_image(img_path)
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

BATCH_SIZE=10
path = str("./data/raw")

transform= None
train_set = CustomImageDataset(img_dir=path, transform=transform, target_transform=None)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

###TRAIN_SET-> sample = one image
print("-----------TRAIN_SET------------")
print("number of all images in all barches: ",len(train_set))
print("name of images and label (our future coordinates od the center/ loss?): \n",train_set.img_labels[:30])
sample=next(iter(train_set))
images, label = sample
print("shape: ",images.shape)

###TRAIN_LOADER-> sample = one batch of images 
print("-----------TRAIN_LOADER------------")
sample=next(iter(train_loader))
images, label = sample
print("number of all images in one barches: ",len(images))
print("shape: ",images.shape)

for index, (images, label) in enumerate(train_loader):
    print("number of all images in " + str(index) +". barch: ",len(images), "-> shape: ", images.shape)
    


grid = torchvision.utils.make_grid(images, nrow=round(BATCH_SIZE/2))
plt.imshow(np.transpose(grid,(1,2,0)))
plt.show()