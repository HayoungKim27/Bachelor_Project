import os
import pickle
from torchvision import transforms
from skimage import io
from skimage.transform import resize

def preprocess_dataset():
	
	train_path = '/home/haykim/dataset/train_images/1/'
	train_images = os.listdir(train_path)
	#test_images = os.listdir('test_images/')

	transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
	#transform까지 전부 하는걸로 진행합시다. 

	for image_name in train_images:
		#code.interact(local=dict(globals(), **locals()))
		try:
			image = io.imread(train_path + image_name)
			#image = cv2.imread(train_path + image_name)
			#image = load_image(train_path + image_name)
			image = resize(image,(224,224))

			#image = transform(cv2.resize(image,(256,256)))
			io.imsave('train_images_crop_224_224/' + image_name, image)

			#code.interact(local=dict(globals(), **locals()))

			#with open('train_images_crop/' + image_name[:-4]+".png","wb") as f:
				
				#pickle.dump(image, f)
			"""for image_name in test_images:
							image = load_image('test_images/'+image_name)
							image = transform(cv2.resize(image,(256,256)))
					
							with open('test_images_pickle/' + image_name+".pkl","wb") as f:
								pickle.dump(image, f)"""
		except:
			print(image_name)

preprocess_dataset()
#transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)), transforms.ToTensor()])
#train_dataset = aptos_dataset("train_images_pickle","train.csv","train",transform)
#code.interact(local=dict(globals(), **locals()))
#image_id, image, label = train_dataset[110]
#merge_chunks_to_train_val('label_chunk')