---

# Behavioral Cloning

---

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, I have used my knowledge on deep neural networks and convolutional neural networks to clone driving behavior. I have trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

Udacity has provided a simulator where you can steer a car around a track for data collection. The image data and steering angles are used to train a neural network and then the trained model drives the car autonomously around the track.


## Goals

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarise the results with a written report


[//]: # (Image References)

[data_visualisation]: ./images/data_visualisation.png "Data Processing"
[model_architecture]: ./images/nvidia_conv_network.JPG "Nvidia Model Architecture"
[learning_graph]: ./images/nvidia_conv_network.jpg "Learning Loss Graph"
[centre]: ./images/centre.jpg "Centre Image"
[left]: ./images/left.jpg "Left Image"
[right]: ./images/right.jpg "Right Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md this report summarising the procedure and results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a 5 layer convolution neural network, consisting of three layers 5x5 filter sizes and 2x2 strides followed by two layers with 3x3 filter sizes and 1x1 strides. The depths for the five layers being 24, 36, 48, 64 and 64 respectively (model.py, model.py lines 105-109). 

Before the convolutional layers the images are first cropped (model.py, line 103) to reduce non-important features for model and then normalised using a Keras Lambda Layer (model.py, line 104).

Each convolutional layer uses RELU activations to introduce nonlinearity, the convolutional layers is followed by Max Pooling to help reduce over-fitting, then five dense layers (model.py, lines 112 to 116) with depths of 1164, 100, 50, 10, 1 respectively. 

#### 2. Attempts to reduce overfitting in the model

The model was tested using dropout layers, however it was found the model was not overfitting using only one Max Pooling layer (model.py line 110). Additional dropout layers showed worse training and validation results, perhaps due to the limited data used.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py, lines 2). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track in both forward and reverse directions. No data was collected for reverse driving in order to provide new test data. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118). Epoch of 10 was typically used and a batch size of 100. The batch size increases intraining due to data augmention within the model pipeline. 

Epoch 1/12
242/241  - 188s - loss: 0.0401 - val_loss: 0.0286
Epoch 2/12
242/241  - 178s - loss: 0.0280 - val_loss: 0.0238
Epoch 3/12
242/241  - 191s - loss: 0.0236 - val_loss: 0.0238
Epoch 4/12
242/241  - 159s - loss: 0.0217 - val_loss: 0.0240
Epoch 5/12
242/241  - 123s - loss: 0.0205 - val_loss: 0.0200
Epoch 6/12
242/241  - 167s - loss: 0.0191 - val_loss: 0.0203
Epoch 7/12
242/241  - 174s - loss: 0.0196 - val_loss: 0.0225
Epoch 8/12
242/241  - 123s - loss: 0.0178 - val_loss: 0.0189
Epoch 9/12
242/241  - 115s - loss: 0.0178 - val_loss: 0.0184
Epoch 10/12
242/241  - 114s - loss: 0.0170 - val_loss: 0.0166
Epoch 11/12
242/241  - 115s - loss: 0.0163 - val_loss: 0.0167
Epoch 12/12
242/241  - 125s - loss: 0.0175 - val_loss: 0.0186

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, as well as driving straight on either side of the road. By driving straight toward either edge of the road helped to prevent occillations around the roads midpoint.

It was interesting to note that when using a small training set with carefully selected data, a much lower mse it acheived, however running the simulation would perform worse, i.e.
Note: new training data set collected and trained
73/72 [==============================] - 74s - loss: 0.0180 - val_loss: 0.0065
Epoch 2/5
73/72 [==============================] - 67s - loss: 0.0047 - val_loss: 0.0040
Epoch 3/5
73/72 [==============================] - 56s - loss: 0.0033 - val_loss: 0.0034
Epoch 4/5
73/72 [==============================] - 54s - loss: 0.0031 - val_loss: 0.0028
Epoch 5/5
73/72 [==============================] - 55s - loss: 0.0027 - val_loss: 0.0028

Following this result the original data set was used with additional collection. Read on for more details on training data collection.

## Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with an existing architecture which could be setup quickly and is known to work on images feature recogonition. Being able to setup up the model quickly and begin testing was important to ensure the entire process was working. 

My first step was to use a single 3x3 filter size convolutional layer with no processing to the images. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error (~94) on the training set and validation set. Despite this I test the model on the simulation and found the car was able to drive, however it quickly veered of the edge of the road.

My next step was to add normalisation to the images, this brought the error down to 1.7 for the test data and 6 for the validation set. The error is much lower however is is overfitting. Testing the model with the car simulation showed some improvement however the car still swerved off the edge soon after starting. 

My next step was changing the model architecture. The LeNet architecture was implemeneted as it is well known to identify features in images. This brought the training accuracy down to 0.0072 and the validation accuracy to 0.0208, still over fitting. It was also noticed that the car had a tendency to turn left on the track. This is likely due to the collected data favouring the anti-clockwise direction.

The next step was to add data augmentation and processing to the model. I found I ran out of memory on my GPU so I used a generator to do the augmentation on the GPU during model training for each batch. To augment the data all of the images were flipped horizontally. The steering data must also be inverted by multiply by -1 to ensure the correct steering direction. The images were also cropped to remove features in the images which may distract the model, such as trees, sky, mountains etc. 65px from the top edge and 22px from the bottom edge were found to remove most distractions. 

There was no noticeable difference with the training and validation accuracy however the car was now driving much better with less tendency to drive of the track.

The next step was adding left and right images from the simulation car for recovery. These images provide more information on the cars state and and can provide an added effect in the steering angle. This was found to to significant increase the ability to steer away from edges. Initialy a steering corection componentent of 0.5 was used however the with more testing and increased car speed this was reduced to 0.1 to prevent over steering. 

To combat the overfitting and improve the model I implemented a different model architecture as used by Nvidia, the model architect can be seen in the image below. This model did not overfit, this is likely due to the max pooling layer used. Here is a visualization of the Nvidia architecture from http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
![alt text][model_architecture]

While the model did not show any improvement in training or validation accuracy, there was noticeable improvement in the simulation. Most of the track could be completed, however at times, particularly corners, the car would drive off the track. Other times the car would begin oscillating on a straight road and eventually hit a verge. It was found collecting more runs at different positions on the road significantly reduced oscillations. 

With some more training data the vehicle was able to drive autonomously around the track without leaving the road, both in clockwise and anti-clockwise directions.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-116) consisted of a convolution neural network with the following layers and layer sizes 
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 73, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 73, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 69, 316, 8)        608
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 33, 156, 24)       4824
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 76, 36)        21636
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 36, 48)         43248
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 34, 64)         27712
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 2, 32, 72)         41544
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 1, 16, 72)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1000)              1153000
_________________________________________________________________
dense_2 (Dense)              (None, 100)               100100
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,398,243
Trainable params: 1,398,243
Non-trainable params: 0
_________________________________________________________________


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, one lap on track with the car offset from the centre toward the left edge and one lap with the car offset from the cetre to the right edge. I recorded data some sections with a gradual correction from the edges to the centre, stopping the recording before the centre is reach. I found if bad images were collected it affected the outcome of the model on the simulation, there was no easy way to filter out bad behaviours without manually removing this from the data set. 

Using the centre, left and right images helped in recovery.

To augment and process the data set, I flipped images and angles to prevent left track bias.  After the augmentation, I experimented with grayscaling images, CLAHE normalisation and cropping. My final solution uses RGB and Keras Normaliation Layer.

Images showing processing techniques:
![alt text][data_visualisation]

I randomly shuffled the data set (model.py, line 75) and put 20% of the data into a validation set (model.py, line 22). A separate test data set was not used as I figured the simulation environment would provide sufficient test data, particularly in the reverse direction and on Track 2. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the graph of learning loss below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][learning_graph]

#### 4. Results

The resulting autonomous lap of Track 1 in can be seen at https://youtu.be/H_HG4i3yUzA 

Videos from the onboard cameras of the autonomous car simulation running in clockwise and anticlockwise can be found in run1.mp4 and track1reverse1.mp4 of the project files in this repository.
