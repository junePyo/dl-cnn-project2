import cv2
import os
import glob
import torch
import random

torch.set_default_dtype(torch.float32) #set the default Torch datatype to 32-bit float

img_dir = 'face_images'
files = glob.glob(os.path.join(img_dir, '*.jpg'))
data = []

#data augmentation functions
def horizontal_flip(img):
    return cv2.flip(img, 1)

def random_crop(img):
    h, w = img.shape[:2]
    crop_size = random.randint(64, 127)
    i = random.randint(0, h-crop_size)
    j = random.randint(0, w-crop_size)
    return img[i:i+crop_size, j:j+crop_size]

def scaling(img):
    factor = random.uniform(0.6, 1.0)
    return img*factor

for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (128, 128)) #height and width are both 128

    #original image
    data.append(torch.tensor(img.copy()))

    #data augmentation - 1 group only
    data.append(torch.tensor(horizontal_flip(img.copy())))
    data.append(torch.tensor(random_crop(img.copy())))
    data.append(torch.tensor(scaling(img.copy())))

    #data augmentation - 2 groups
    data.append(torch.tensor(horizontal_flip(random_crop(img.copy()))))
    data.append(torch.tensor(horizontal_flip(scaling(img.copy()))))
    data.append(torch.tensor(scaling(random_crop(img.copy()))))

    #data augmentation - 1 group
    data.append(torch.tensor(horizontal_flip(scaling(random_crop(img.copy())))))

    #2 groups of random scaling to fill 10x tensor
    data.append(torch.tensor(scaling(img.copy())))
    data.append(torch.tensor(scaling(img.copy())))

os.makedirs('augmented', exist_ok=True)
for i, img in enumerate(data):
    cv2.imwrite(f'augmented/image_{i}.png', img.numpy())

tensor_data = torch.stack(data) #Load your data in a Tensor
shuffled_data = tensor_data[torch.randperm(tensor_data.size(0))] #randomly shuffle the data using torch.randperm

#L*a*b
os.makedirs('L', exist_ok=True)
os.makedirs('a', exist_ok=True)
os.makedirs('b', exist_ok=True)

data_lab = []
l_channel = []
a_channel = []
b_channel = []

for img in data:
    img = img.numpy()
    img = cv2.convertScaleAbs(img)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data_lab.append(img_lab)

for img in data_lab:
    l, a, b = cv2.split(img)
    l_channel.append(l)
    a_channel.append(a)
    b_channel.append(b)

for i, (l, a, b) in enumerate(zip(l_channel, a_channel, b_channel)):
    cv2.imwrite(f'L/image_{i}.png', l)
    cv2.imwrite(f'a/image_{i}.png', a)
    cv2.imwrite(f'b/image_{i}.png', b)
