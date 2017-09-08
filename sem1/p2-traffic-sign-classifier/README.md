## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project uses my knowledge on deep neural networks and convolutional neural networks to classify traffic signs. In this project I train and validate a model so that it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model has been trained, I evaluate the model on several images of German traffic signs found from the web.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarise and visualise the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyse the softmax probabilities of the new images
* Summarise the results with a written report


[//]: # (Image References)

[image1]: ./plots/data_bins.png "Visualization"
[image3]: ./plots/augmentations_yield.jpg "Augmented Data"
[image4]: ./plots/augmentations_yield.png "Traffic Sign 1"
[image5]: ./plots/preprocessing_nopassing.pngg "Preprocessing Sign 1"
[image6]: ./plots/augmentations_nopassing.png "Augmentation Sign 1"
[image7]: ./plots/preprocessing_gostraightorright.png "Preprocessing Sign 2"
[image8]: ./examples/placeholder.png "Augmentation Sign 2"
[image9]: ./plots/preprocessing_speedlimit120kmh.png "Preprocessing Sign 3"
[image10]: ./examples/augmentations_speedlimit120kmh.png "Augmentation Sign 3"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/revspete/self-driving-car-nd/blob/master/sem1/p2-traffic-signs/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 34799.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3.
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualisation of the data set. It is a bar chart showing how many images there are for each of the 43 labels. It is clear that the data is not evenly balanced across the labels, the range of data samples per label being between 180 to 2320. This can lead to poor training of some labels in comparison to others.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce the number of layers that would be processed. I tried running the model on both RGB and Grayscale images and found no improvement with RGB, in fact generally the model was not as suited to the RGB as the grayscale. Futhermore the grayscale images were faster to train with.

Here are some examples of a traffic sign images under going some of the various pre-processing cases including gray scale, normalisation, sharpening, de-noise and sobel filters. 

![alt text][image5]
![alt text][image7]
![alt text][image9]


I tried several different techniques for the normalisation and found the preprocessing scale function most consistent / reliable. Other normalisation methods shows significant white or black spots within the images at times. 

Sharpening the image at times presented visually better images, and other times less visually plesant. In general sharpening the images added a percentage improvement in accuracy.

I decided to generate additional data because I was worried the model would bias training for those with more data. When checking validation accuracy it will be worth noting which images have lower accuracy and what correlation there is to the number of images originally and the affect on data augmentation. 

To add more data to the the data set, I performed a series of transformations including rotations, translations and shear. The amount of rotation, translation or shear was determined randomly, however the amount of transformation was limited so that the images were not confused. It is unlikely to find signs upside down, back to front and only half in image. To prevent stack multiple transformations on top of another only the original data set is used for augmentation. The number of images per bin can be specificed. I tried add noise and blurring images, however I found many images were already very blurred and noisy and I did not want to futher affect the image quality. For a given model I tried increasing the number of augmented data images for each bin to 2000, 2500, 5000 and 10000. I found an improvement from 2000 to 2500, however not futher improvements in accuracy to 5000 or 10000.

Here is an example of an original image and an augmented image:

![alt text][image3]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding					 	|
| DROPOUT				| 0.5											|
| RELU					|												|
| Av pooling	      	| 2x2 stride									|
| Convolution 5x5	    | etc.     Depth Output 32 						|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I initally used trial and error with the LeNet architecture as basis.

Modifiying one parameter at I time I observed changes in accuracy. Plotting the training and validation accuracy was important in understanding whether the model was underfitting or over fitting and how fast it was performing.

I found large batch sizes would smooth the learning curve and speed the training, however at times it would also reduce the accuracy and cause over fitting.

I typically set Epoch values to 100+, if I noticed it had converged I would stop training.

I experimented with dropouts and found most models were over fitting so they were applied. The dropout were more benifical for the more complex inception models. In the Lenet model it would reduce accuracy and not prevent over fitting.

I choose the Adam optimiser based on discussions and reccommendations in the community. I have not run the models with other optimisers yet. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The LeNet architecture was chosen first following testing with the MNIST training. It was easy to modify for the 32x32 gray scale images and run with descent accuracy.

* What were some problems with the initial architecture?
It was difficult to understand the best way to modify the architecture to make improvements for the traffic signs.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
One of the major issues with the project so far has been overfitting. Following research into how to prevent over fitting, a commmon approach is to apply drop out after each layer. It was learnt that using a lower drop out rate for the early convolution layers and a high drop out rate for the later layers can prevent the model from failing.
It was found that making the depth deeper and adding a convolution layer increased the accuracy most. Unfortunately adding the extra layer and depth also increase overfitting.

* Which parameters were tuned? How were they adjusted and why?
Following each change in architecture, changes to the learning rate were often performed, particularly if the learning curve was jumpy in nature.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are a number of German traffic signs that I found online:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some of these images will obviously be difficult to classify as they are not amongst the labels being trained. However I thought it would be interesting to see what they are classified as. I suspect these would be predicted with high accuracy as they are are icons with good quality. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 11 of the 16 classfied traffic signs, which gives an accuracy of 68.7%. I expected this to be greater due to the quality given. Perhaps because only real images were trained, these icons are not identifiable. In some photos I beleive have blue and red channels would be benifical. These colours make a large difference in sign classfication being a "do" or "don't" sign. Some of the incorrectly identify signs have somewhat resemblance and some of the second highest probabilties are are correct. However it should be noted that the first probabilities are very high in comparison to others, I suspect this is due to the dropouts, relu and softmax reducing the probabilities. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)



For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


