import os, code, cv2, PIL
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import imageio
from torch.autograd import Variable
from PIL import Image
from skimage import io 
from models_vgg import vggnet16, vggnet19
from model_alexnet import alexnet 

import code
import pandas as pd
from Test_dataset import TestDataset
from CIFAR10Policy import CIFAR10Policy
from SubPolicy import SubPolicy

import model

from torch.utils.data import Dataset, DataLoader
import torchvision as vision
from pathlib import Path
import csv
from torch import nn, cuda


row_list = [['img_file', 'True_label', 'Predicted_label', 'Probability']]
def create_CAM(data, model, result_path, i):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	features_blobs = []
	def hook_feature(module, input, output):
		features_blobs.append(output.cpu().data.numpy())
	#code.interact(local=dict(globals(), **locals()))
	model._modules.get('layer4').register_forward_hook(hook_feature)

	# get the softmax weight
	params = list(model.parameters())
	weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

	def returnCAM(feature_conv, weight_softmax, class_idx):
		# generate the class activation maps upsample to 256x256
		size_upsample = (224, 224)
		bz, nc, h, w = feature_conv.shape
		output_cam = []
		for idx in class_idx:
			cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
			cam = cam.reshape(h, w)
			cam = cam - np.min(cam)
			cam_img = cam / np.max(cam)
			cam_img = np.uint8(255 * cam_img)
			output_cam.append(cv2.resize(cam_img, size_upsample))
		return output_cam

	classes = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative DR'}
	###img
	image_tensor, label = data[0].type(torch.float), data[1]
	#convert image tensor to image
	DRimage = image_tensor[0].clone().cpu().numpy()   # convert from GPU to CPU
	DRimage = DRimage.transpose((1,2,0))   # convert image back to height, weidth, channels
	imageio.imwrite(result_path+'/img%d.jpg' % (i + 1), DRimage)

	#code.interact(local=dict(globals(), **locals()))
	###heatmap
	logit = model(image_tensor)			
	h_x = F.softmax(logit, dim=1).data.squeeze()
	probs, idx = h_x.sort(0, True)
	print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))
	probs = probs.cpu().numpy()
	idx = idx.cpu().numpy()
		
	CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

	# render the CAM and output
	print('The top1 prediction: %s' % classes[idx[0].item()])
	img = cv2.imread(result_path+'/img%d.jpg' % (i + 1))
	height, width, _ = img.shape

	heatmap = cv2.resize(CAMs[0],(width, height))


	plt.figure()
	plt.imshow(img)
	plt.imshow(heatmap, cmap='jet', alpha=0.5)
	plt.colorbar()
	plt.axis('off')
	plt.savefig(result_path+'/heatmap%d.jpg' % (i + 1))

	###CAM
	DRimage_pillow = cv2.imread(result_path+'/img%d.jpg' % (i + 1))
	DRimage_bgr = cv2.cvtColor(DRimage_pillow, cv2.COLOR_RGB2BGR)
	plt.figure()
	fig, [ax1, ax2] = plt.subplots(ncols=2)
	ax1.imshow(DRimage_bgr)
	ax1.axis('off')
	ax1.set_title(classes[label.item()])
	ax2.imshow(img)
	ax2.imshow(heatmap, cmap='jet', alpha=0.5)
	ax2.set_title(classes[idx[0].item()]+'\n(Prob: %.2f)' % probs[0].item())
	ax2.axis('off')

	plt.savefig(result_path+'/CAM%d.jpg' % (i + 1))	
	
	row_list.append(['img%d.jpg' % (i + 1), label.item(), idx[0].item(), probs[0].item()])

	

########## Visualization using Class Activation Mapping ##########
fold_num = 3 #change
#code.interact(local=dict(globals(), **locals()))
df = pd.read_csv("image_5folds.csv")
test_df = df.loc[df['fold'] == fold_num]
test_df = test_df[['img_file', 'class']].reset_index(drop=True)


target_size = (224, 224)

data_transforms = {
    'train': vision.transforms.Compose([
        vision.transforms.Resize(target_size),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.RandomRotation(20),
        CIFAR10Policy(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'valid': vision.transforms.Compose([
        vision.transforms.Resize(target_size),
        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),
        vision.transforms.RandomHorizontalFlip(),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
    'test': vision.transforms.Compose([
        vision.transforms.Resize((224,224)),
        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),
        vision.transforms.ToTensor(),
        vision.transforms.Normalize(
            [0.485, 0.456, 0.406], 
            [0.229, 0.224, 0.225])
    ]),
}
batch_size = 1
test_dataset = TestDataset(test_df, mode='test', transforms=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model = CNN()     # AlexNet
#model = vggNet16()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet152(pretrained=True)
model.fc = nn.Linear(2048, 5)
cnn = model.to(device)
#model = vggnet16()
#model = vggnet19()


result_path = '/home/haykim/dataset/CAM_result/CAM_result_resnet152'
best_model = '/home/haykim/dataset/best_model_resnet152_fold3.pt' #change

pretrained_dict = torch.load(best_model)
model_dict = cnn.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
cnn.load_state_dict(model_dict)
#model.load_state_dict(torch.load(best_model))

#change the model to evaluation mode
cnn.eval()

for i, data in enumerate(test_loader):
	data[0], data[1] = data[0].cuda(), data[1].cuda()
	with open('CAM_resnet152_fold{}.csv'.format(fold_num), 'w', newline ='') as file: #change cv file
		writer = csv.writer(file)
		writer.writerows(row_list)
	#code.interact(local=dict(globals(), **locals()))
	if i in range(600):
		pass
	elif i in range(600,900):
		create_CAM(data, cnn, result_path, i)
	else:
		break




