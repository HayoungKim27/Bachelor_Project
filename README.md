There are nine python codes(.py) and two jupyther notebook documents with the python language(.ipynb) in the github link.

Resizing_dataset.py resizes the images into an equal size, 224×224 pixels in width and height. fold_split.py splits the initial dataset into five folds for 5-fold cross-validation and makes a CSV file having the columns of image name, class, and fold.

train.py, Test_dataset.py, SubPolicy.py, and CIFAR10Policy.py are used for training the model.

model_alexnet.py, models_vgg.py, Create_CAM_Final.py, and Test_dataset.py are used for creating class activation mapping(CAM). (The first codes are for extracting CAM using model AlexNet and VGGNet. In this study, only the model ResNet was used for extracting CAM since ResNet152 was the best model for the dataset.)

cm_plot_jupyter.ipynb creates the confusion matrix and Transforms vs Albumentations.ipynb calculates the image processing time when the transform from torchvision or the open-source library albumentations is used.
