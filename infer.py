%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

data_dir = '/Data/MoldPID/top/train-full'

test_transforms = transforms.Compose([#transforms.Resize(224),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('pintopmodel-full.pth')
model.eval()
model

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index  

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels
	
import matplotlib as mpl
to_pil = transforms.ToPILImage()
images, labels = get_random_images(256)
fig=plt.figure(figsize=(15,30))
for ii in range(len(images)):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(16, len(images) / 16, ii+1)
    res = int(labels[ii]) == index
    fontParams = {'fontsize': mpl.rcParams['axes.titlesize'],
                  'fontweight': mpl.rcParams['axes.titleweight'], 
                  'color': 'green',
                  'verticalalignment': 'baseline', 
                  'horizontalalignment': 'center'}
    if res == False:
        fontParams['color'] = 'red'
        print(fontParams)
    sub.set_title(str(classes[index]) + ":" + str(res), fontdict=fontParams)
    plt.axis('off')
    plt.imshow(image)
plt.show()

