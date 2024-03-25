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
    i = random.randint(0, h-128)
    j = random.randint(0, w-128)
    return img[i:i+128, j:j+128]

def scaling(img):
    factor = random.uniform(0.6, 1.0)
    return img*factor

for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img, (128, 128)) #height and width are both 128

    #data augmentation
    augmentations = [horizontal_flip, random_crop, scaling]
    random.shuffle(augmentations)
    for augmentation in augmentations:
        img = augmentation(img)
        data.append(torch.tensor(img))

os.makedirs('augmented', exist_ok=True)
for i, img in enumerate(data):
    cv2.imwrite(f'augmented/image_{i}.png', img.numpy())

tensor_data = torch.stack(data) #Load your data in a Tensor
shuffled_data = torch.randperm(tensor_data.size(0)) #randomly shuffle the data using torch.randperm
new_data = tensor_data[shuffled_data]

#L*a*b
os.makedirs('L', exist_ok=True)
os.makedirs('a', exist_ok=True)
os.makedirs('b', exist_ok=True)

data_lab = []
L_channel = []
a_channel = []
b_channel = []

for img in data:
    img = img.numpy()
    img = cv2.convertScaleAbs(img)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data_lab.append(img_lab)

for img in data_lab:
    L, a, b = cv2.split(img)
    L_channel.append(L)
    a_channel.append(a)
    b_channel.append(b)

for i, (L, a, b) in enumerate(zip(L_channel, a_channel, b_channel)):
    cv2.imwrite(f'L/image_{i}.png', L)
    cv2.imwrite(f'a/image_{i}.png', a)
    cv2.imwrite(f'b/image_{i}.png', b)

cv2.imshow('Image', data[0].numpy().astype('uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('L', L_channel[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('a', a_channel[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('b', b_channel[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
