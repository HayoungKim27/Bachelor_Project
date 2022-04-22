# Bachelor Project: Automated Early Detection of Diabetic Retinopathy in Retinal Fundus Photographs using Deep Learning
Welcome to the 'Bachelor Project: Automated Early Detection of Diabetic Retinopathy in Retinal Fundus Photographs using Deep Learning' repository!
This is the Bachelor graduation project of Ghent University.



## Guideline
There are nine python codes(.py) and two jupyther notebook documents with the python language(.ipynb) in the folder **py** and **ipynb** respectively.
The entire procedure is divided into 5 steps.

1. Image preparation
2. Split in 5 Folds
3. Image Classification
4. Visualization

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/entire_pipeline.png" width="60%" height="60%">

### Image Preparation
_Resizing_dataset.py_ resizes the images into an equal size, 224×224 pixels in width and height.

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/resizing.png" width="40%" height="40%">

(a) A sample image showing the retina partially (2416 x 1736)
(b) Resized image of (a) (224 x 224)
(c) A sample image showing the retina entirely (2048 x 1536)
(d) Resized image of (c) (224 x 224)

### Split in 5 Folds
_fold_split.py_ splits the initial dataset into five folds for 5-fold cross-validation and makes a CSV file having the columns of image name, class, and fold.

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/5folds.png" width="50%" height="50%">

### Image Classification
_train.py, Test_dataset.py, SubPolicy.py_, and _CIFAR10Policy.py_ are used for training the model.

### Visualization
_model_alexnet.py, models_vgg.py, Create_CAM_Final.py,_ and _Test_dataset.py_ are used for creating class activation mapping(CAM).
(The first codes are for extracting CAM using model AlexNet and VGGNet. In this study, only the model ResNet was used for extracting CAM as ResNet152 was the best model for the dataset.)

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/model_alexnet.png" width="22%" height="22%">

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/model_vgg16.png" width="40%" height="40%">

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/model_vgg19.png" width="45%" height="45%">

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/model_resnet.png" width="60%" height="60%">

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/plainlayers_vs_residualblock.png" width="40%" height="40%">

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/finetuning.png" width="50%" height="50%">

### Creating the confusion matrix and calculating the image preprocesing time
_cm_plot_jupyter.ipynb_ creates the confusion matrix and Transforms vs Albumentations.ipynb calculates the image processing time when the transform from torchvision or the open-source library albumentations is used.

## Results
### Quantitative results
#### Model Comparison
<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/f1score.png" width="60%" height="60%">

The micro-averaged F1 score was the highest when the dataset was trained with the ResNet152 model. In particular, the highest micro-averaged F1 score was obtained when Fold 3 was used as the test set. Moreover, it can be seen that the standard deviation for each model is all very low, indicating that the microaveraged F1 scores of each fold are closer to the mean.

#### Performance Metrics on the Best Model

When the selected best model is evaluated with Fold 3 as the test set, the confusion matrix is as follows.

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/confusion_matrix.png" width="40%" height="40%">

### Qualitative results
Among the results of the CAM for the best performing ResNet152 model, those following two results are 'class 2: Moderate' and 'class 3: Severe' respectively. The parts marked with a red ellipse are regions having important features when DR is classified into each class.

The description of the features are in the discussion section of the paper.

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/moderate.png" width="50%" height="50%">

<img src = "https://github.com/HayoungKim27/Bachelor_Project/blob/master/image/severe.png" width="50%" height="50%">

