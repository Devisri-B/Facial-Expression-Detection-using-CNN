# Facial Expression Detection using CNN

## Description
This project focuses on detecting facial expressions using a Convolutional Neural Network (CNN). The model is trained to recognize various facial expressions such as happiness, sadness, fear, surprise, neutrality, anger, and disgust from image data. The dataset used for training and evaluation is sourced from Kaggle's Face Expression Recognition Dataset. https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

## Project Overview
Facial expressions are a vital form of non-verbal communication. The goal of this project is to classify images of human faces into one of several expression categories using deep learning techniques.

## Dataset Structure
The dataset used for this project is the Face Expression Recognition Dataset from Kaggle. It contains a wide range of facial images labeled with corresponding expressions, making it suitable for training deep learning models for emotion recognition.
- train: Used to train the CNN model.
- validation: Used for evaluating the model's performance during training.
#### Classes
The dataset consists of the following facial expression classes:
Happy,
Sad,
Fear,
Surprise,
Neutral,
Angry,
Disgust,
Image Counts

### Image Counts
#### Training Images:
7,164 happy images, 
4,938 sad images,
4,103 fear images,
3,205 surprise images,
4,982 neutral images,
3,993 angry images,
436 disgust images
#### Validation Images:
1,825 happy images,
1,139 sad images,
1,018 fear images,
797 surprise images,
1,216 neutral images,
960 angry images,
111 disgust images,

## Model Architecture
The model uses a Convolutional Neural Network (CNN) to process and classify the input images. The architecture is designed to capture spatial hierarchies and complex features from the images through layers of convolutional, pooling, and fully connected layers.

### Data Augmentation
Data augmentation techniques used include:
Image rescaling, 
Random rotations,
Width and height shifts,
Shear transformations,
Zoom operations,
Horizontal flips,
These augmentations help to enhance the model's generalization capability.

## Usage
### Requirements
Python 3.x,
TensorFlow,
Keras,
OpenCV,
NumPy,
Matplotlib,
Jupyter Notebook (recommended for running the provided notebook)

## Data Preparation
Ensure that the dataset is organized in the following structure:
```markdown
dataset/
    train/
        happy/
        sad/
        fear/
        surprise/
        neutral/
        angry/
        disgust/
    validation/
        happy/
        sad/
        fear/
        surprise/
        neutral/
        angry/
        disgust/
```

## Training the Model
Run the Jupyter Notebook Facial_expression_detection_using_CNN.ipynb to train the model. The notebook covers data loading, preprocessing, model building, training, and validation steps.

## Evaluating the Model
The model's performance can be evaluated using the validation set. Metrics such as accuracy, precision, recall, and loss can be monitored to gauge the model's effectiveness.

## Results
During the training and validation process, you can expect to observe model accuracy and loss metrics. The model's ability to recognize facial expressions depends on various factors, including data quality and model hyperparameters.



