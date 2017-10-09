# Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Explore color transforms, binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalise  features and randomise a selection for training and testing.
* Implement a sliding-window technique and use a trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_not_car]: 		./output_images/car_not_car.png
[hogfeatures]: 		./output_images/HOG_example.png
[normalised]: 		./output_images/normalised_features.png
[searchregion]: 	./output_images/search_region.png
[bboxes]: 			./output_images/bboxes.png
[heatmap]: 			./output_images/heatmap.png
[labels]: 			./output_images/labels.png
[filtered]: 		./output_images/filtered.png
[image8]: 			./output_images/output_bboxes.png
[img1]: 			./output_images/img1.png
[img2]: 			./output_images/img2.png
[img3]: 			./output_images/img3.png
[img4]: 			./output_images/img4.png
[img5]: 			./output_images/img5.png
[img6]: 			./output_images/img6.png
[test1]:			./output_images/test1.png
[test2]:			./output_images/test2.png
[test3]:			./output_images/test3.png
[test4]:			./output_images/test4.png
[video1]: 			./project_video.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook 'p5-Vehicle-Detection-And-Tracking'.  

The code cell also includes methods to convert the colour of the image, bin the colour features, get a histogram of colour fgeatures and get the HOG features. 

In cell three I start by reading in of the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

```
No. images with cars 8792
No. images without cars 8968
```

![Vehicle vs Not Vehicle][car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

```
Extracting Car Features
74.66 Seconds to extract HOG features...
Car Features Shape:  8792

Extracting No Car Features
70.87 Seconds to extract No Car HOG features...
No Car Features Shape:  8968
```

![HOG Features][hogfeatures]

When combing different features such as colour histogram features and HOG features it is important that the features are normalised, this is done with StandardScaler from sklearn and can be seen in the following image:

![Normalised Features][normalised]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and colour channels, and observed the speed of eature extraction and accuracy in my trained classifier.

Colour Space: By only changing the colour space I found that using all channels of the `LUV` gave the highest test accuracy. 

Orient: I found that increasing the orient reduced accuracy and made the feature extraction slower. On the otherhand small orients were also less accurate. I found `9` to have the highest accuracy.

Pixels per cell: Increasing this parameter sped feature extraction time up, however I found `8` to be the most accurate.

Cells per block: Higher cell count was faster, however reduced accuracy. I found `2` to be most accurate.

HOG Channel: I looked at individual channels however using all channels was most accurate.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC() from sklearn SVM using an 80% training set and 20% test set. I was able to acheive a high with 0.003 seconds per prediction which seemed reasonable in speed. The linear SVM training is in code cell 9. The features included LUV colour binning, colour histogram features, and HOG features. 

```
Splitting Train and Test sets
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8412
26.0 Seconds to train SVC...
Test Accuracy of SVC =  0.9938
My SVC predicts:  [ 1.  1.  0.  1.  1.  0.  1.  0.  0.  0.]
For these 10 labels:  [ 1.  1.  0.  1.  1.  0.  1.  0.  0.  0.]
0.03359 Seconds to predict 10 labels with SVC
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding window approach has been implemented, where overlapping tiles in each test image are classified as a vehicle or non-vehicle. To improve efficiency a single function `find_cars()` uses a HOG sub-sampling method to extract features once across a designated search window and then make predictions. To futher improve speed my final implementation searches only for cars that are entering from the left, right sides of the image, or the horizon. I create three search spaces, one for each region as seen here:

![Search Region][searchregion]

For each region I extract the features and predict bounding boxes for vehicles. Each region uses a different scale factor as the expect car will vary in size. The horizon uses a scale factor of 0.5, whilst the sides use a scale factor of 1.4: 

![Bounding Boxes][bboxes]

Once a car has been identified a search window of 200 x 200 around the cars next predicted location is used. For this search area three scale factors are used being 1, 1.5 and 2. This helps improve the chances of tracking the vehicle at all locations in the image and adds to the weight of its heatmap. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


To optimise the performance of the classifier I tried numerous scenarios/frames from the project video. It can be noted that very similar scenes can have a surprising difference in the classification, leading to false positives and false negatives. Averaging is important to reject the false positives and to ensure the actual objects are not lost in tracking. Here a a few examples:

![Test 1][test1]

![Test 2][test2]

![Test 3][test3]

![Test 4][test4]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/n7I_I_-NGGw)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to improve the likelihood of a vehicle detection several techniques are used.

Heat map: I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded map to identify vehicle positions. 10 previous heat maps are stored and weighted together, using 70% of the new frame and `1-index/total` percent for each heat map previous. This helps to reduce noise in new heat maps and prevents vehicles from temporarily disapearing in future images.  A final threshold is applied to the averaged heat map to reduce noise.

![Heat Map][heatmap]

Labels and validity checks: From the averaged and threshold heat map, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map. The position and size of each label is determined and some validity checks are performed i.e. on the size, aspect ratio and location of the predicted car. 

![Labels from Heat Map][labels]

Following the heat map, labeling and validity checks, it can be seen that misclassified vehicles can be removed from most scenes: 

```
Fixed.  Aspect: 0.562 Height: 86 Width: 153 x: 1117.5 y 459.0
FAILED. Aspect: 0.667 Height: 18 Width: 27  x: 399.5  y 428.0
Fixed.  Aspect: 0.651 Height: 82 Width: 126 x: 898.0  y 451.0
```

![Filtered Labels on Image][filtered]

For future frames I store the centre location of each car for the past 5 frames. I then take a linear regression of the centre locations for each car and determine the direction it is moving in x and y coordinates to predict its future position. 

This works in most situations however at times the vehicle label may change i.e. a car label may change from 0 to 1 as it moves across the image. Currently I have not implemented a method to order the labels and check for mislabelling. I expect this could be done with a binning operation. Using the linear regression, we can predict the location of the car and create a new optimsied search window location.   

### Here are six frames and their corresponding heatmaps, averaged and threhold heat maps, labels, validated and filter image with bounding boxes:

![alt text][img1]
![alt text][img2]
![alt text][img3]
![alt text][img4]
![alt text][img5]
![alt text][img6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Robustness and Tracking: When a car passses another car the tracking is lost. I would like to implement a more robust tracking method which will realise there are two vehicles and one is obscured. My pipeline will not handle tracking objects which do not appear from the sides or horizon. i.e. if tracking begins with a car directly in front. On initially sequence I could search entire frame then only search corners. 

Lanes: I would like to overlay the lane lines from the Advanced Lane Finding Project. I could also use the lane lines width and location as start and stop locations for the sliding windows to search for cars in each lane. 

Classifier: I would like to experiment with using a convolutional neural network to classify objects in the image as car or not car, and extend this to other objects in general i.e. people,  bicycles and traffic signs. 

Processing time: I found processing the entire video slow using CPU. The video take 40mins to process 50 seconds of footage or 2 seconds per frame. My first goal will be to make the process run in real time. In order to do this I will time each function in the pipeline to determine the bottle necks. Reducing the number of feature extraction will speed up the process, current I perform 3 feature searches for each car. Firstly this could be reduced and the scale could be related to the position on the image. Secondly the search areas do not consider whether the area has been previously search therefore duplicating areas. Implementing a class to hold the previous locations and averaging along the way may reduce the number of times i need to loop through previous data. 

### References
Many of the methods, videos and images used in this project have been copied and/or adapted from the Udacity Self Driving NanoDegree.