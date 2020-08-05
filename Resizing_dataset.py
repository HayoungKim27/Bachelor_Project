import os
import pickle
from torchvision import transforms
from skimage import io
from skimage.transform import resize

def Resizing_dataset():
	
	train_path = '/home/haykim/dataset/train_images/1/'
	train_images = os.listdir(train_path)

	transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

	for image_name in train_images:
		try:
			image = io.imread(train_path + image_name)

			image = resize(image,(224,224))

			io.imsave('train_images_crop_224_224/' + image_name, image)


		except:
			print(image_name)

Resizing_dataset()
