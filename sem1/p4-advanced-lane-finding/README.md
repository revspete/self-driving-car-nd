## Advanced Lane Finding

### Udacity Self Driving Car NanoDegree Project 4 - Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: 	./output_images/output_undist.png "Undistorted"
[image2]: 	./output_images/lane_output_undist.png "Road Transformed"
[gray]: 	./output_images/gray_thresholds.png "Gray Thresholds"
[sthresh]: 	./output_images/s_thresholds_compare.jpg "S Thresholds"
[s_mag]:	./output_images/s_mag.jpg "S Gradient Magnitude"
[lab]: 		./output_images/lab_thresholds.png "Lab Space"
[combined]: ./output_images/combined.png "Combined Binary"
[lanewarp1]:./output_images/lane1warp.png "Lane Warp Example 1"
[lanewarp2]:./output_images/lane2warp.png "Lane Warp Example 2"
[selectlanes]:./output_images/selectlanes.jpg "Select Lanes"
[histogram]:./output_images/histogram.png "Histogram of Lane Lines"
[windows]:	./output_images/windows.png "Windows of Lane Lines"
[margin]:	./output_images/margin.png "Margin from Lane Lines"
[laneimg]:	./output_images/laneimg.png "Image with Lane Projection"
[video1]: 	./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first few cells of the IPython notebook located in "p4-advanced-lane-detection-playground.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I take an example lane image and apply the same camera_calibration matrix to remove distotrtion. The distortion correction is most noticeable around the edges, for example the location of the car and angle of the car hood.
 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I investigated many combinations of color and gradient thresholds to generate a binary image. For each of the colour channels Gray Scale, Red, HLS, and Lab I applied a range of thresholds for each of the example images provided to ensure minimal noise obtained in the final binary image. Here's some examples of the outputs for the S channel.
![alt text][sthresh]

S channel Thresholds - I found S Channel from HLS was robust to scene and lighting changes however didn't always pick up the white lines.  R Channel on the other hand picked up white lanes well.

Gray Scale Thresholds - I found gray channel was not invariant enough to changes in the scene and lighting, generating noise on the binary image. With a tigheter threshold no more useful information was gained in comparison to other solutions.
![alt text][gray]

Testing LAB channel Thresholds - Whilst investigating the different colour spaces I noted that other experiements  (http://www.learnopencv.com/color-spaces-in-opencv-cpp-python/) demonstrated that colours were not greatly affected by illumination in the LAB space. In my tests I found this to be true for the LAB space.
![alt text][lab]

To gain most lane data and minimise noise I obtained useful thresholds for the HSV:S,RGB:R,LAB:L,LAB:B channels and combined them to form a single binary image. The result can be seen below:
![alt text][combined]

After acheiving the best combined binary for colour channels I investigated the gradient directions and magnitudes of each channel and the combined threshold binary  for each example image. I focused on reducing noise around the lane lines. Dilation and erroision operations were also used to reduce noise on images. It was found taking the inv image and dilating the pixels was useful as a mask to remove noise. This was not used in the final pipeline as a fixed mask and top down view avoided the need for this.

![alt text][s_mag]

In the end I found the best pipeline was simply combining the colour channels with thresholding and a mask as shown above. I suspect futher investigation in direction gradients may be required to tackle the more challenging videos for a more robust solution.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In order to get a Top Down or Birds Eye View for the image I developed a process to allow users to select four images on the lane. By clicking on one side of the lane a red line is draw to help guide the user to select a matching point on the opposite side.
 
![alt text][selectlanes]

The four points are ordered and passed to a function called `corners_unwarp()` which takes the image `img`, the corners `ordered_coords`, and camera calibration parameters `mtx`,`dist`. It then generates the perspective transform matrix `M` and it's inverse `Minv` which can be written to file and recalled as required.

The destination points have been hard coded, based on the size of the image and a defined offset. 

```python
dst = np.float32([[offset, 0], 						## Top Left
                  [img_size[0]-offset, 0], 			## Top Right
                  [img_size[0]-offset, img_size[1]],   ## Botom Right
                  [offset, img_size[1]] 			   ## Bottom Left
                 ])
```

I verified that my perspective transform was working as expected by drawing a filled polygon between the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Example 1 

![alt text][lanewarp1]

Example 2

![alt text][lanewarp2]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 596, 447      | 350, 0        | 
| 684, 447      | 930, 720      |
| 1057, 677     | 930, 720      |
| 253, 677      | 350, 0        |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify lane line pixels in the rectified binary image, I first took a histogram of bottom half of the image to determine the x positions for the left and right lines. 

![alt text][histogram]

| Lane        | X pixel   | 
|:-------------:|:-------------:| 
| Midpoint:      | 640         | 
| Left x base:      | 402      |
| Right x base:   | 953   |


Using the position of the lines, I take a window and determine the number of non-zero pixels in the window. I then slide the window up and recentre on the window on the mean x position.

```python
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    # Draw the windows on the visualization image
    # Identify the nonzero pixels in x and y within the window
    # Append these indices to the lists
    # If you found > minpix pixels, recenter next window on their mean position

# Concatenate the arrays of indices
# Extract left and right line pixel positions
# Fit a second order polynomial to each
```

![alt text][windows]

Having found the lines once, lines in the next image can be found easily, skipping the sliding windows and searching within a margin of the previous lines. 

```python
margin = 100
left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
left_fit[1]*nonzeroy + left_fit[2] + margin))) 

right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
right_fit[1]*nonzeroy + right_fit[2] + margin)))  
```

![alt text][margin]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To estimate the radius of curvature for the lines, we make some assumptions. We know the height of the image is 720px and we can assume that the distance of the road in the image is ~40m.  According to the US design specifications for roads we can assume the road width is 3.7m and the min road curvature radius is 150m. We can calculate the lane width in pixels. With this we can convert image pixels to real world units, metres, and recalculate the polynomial fit.   

```python
lane_width_px = rightx_base-leftx_base
ym_per_pix = 40/720 # meters per pixel in y dimension
xm_per_pix = 3.7/lane_width_px # meters per pixel in x dimension

# Fit a second order polynomial to each
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

We can also determine the position of the car on the lane
```python
centre = (leftx_base + rightx_base)/2
position = (midpoint - centre)*xm_per_pix
```

| Left Lane Curvature Radius        | Right Lane Curvature Radius   |  Position |
|:-------------:|:-------------:|:-------------:|
| 737.7 m      | 676.5 m          |  0.22m to the left |


Based on the location the images were sourced we can cross reference the roads curvature on google maps and we can say that these radius of curvatures are approximately right.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `get_lane()` which takes in the lane image `undst_img`, a processed warped binary `combined_warped_binary` and optionally the current left `Left.current_fit` and right fits `Right.current_fit` and wether or not to plot the images `plot_on`.

The fit from the rectified image has been warped back onto the original image and the lanes are plotted highlighting the lane boundaries hence demonstrating the lane boundaries are correctly identified. Below is an example image with overlays of the lanes, measured curvature, and measured position from center.

![alt text][laneimg]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

A pipeline has been created to take a raw images from a video, find the lane lines and return the lanes image with lane, curvature and position overlays.

Here's some attempts so far:
Medium Trajectory: https://youtu.be/wulyFSmR5Z8
Long Trajectory: https://youtu.be/qfFRg9h1f7s
Short trajectory: https://youtu.be/-GwvK1TGEBQ

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project has taken a lot of time to complete (~40hrs), a lot of time has been in investigating colour spaces, colour thresholds, direction and magnitude of gradients, and combinations of image processing. 

More time should have been spent setting up the pipeline before tuning the image processing pipeline. There is a lot of room for improvement in the code, structure, pipeline, commenting, class implementation. Futher improvements need to be made for shadows, changes in pavement. Images should be collected for these scenarios and thresholds tuned accordingly. 

The challenge videos have not been attempted yet however, extended patches of missing frames, bursts of light, twisty roads, other traffic will likely cause additional issues for th pipeline. 

One solution for smoothing the lane detection and increasing accuracy of polyfit lines was to take n frames and combined the processed binary images. This is useful for situations when there are not enough points on one of the lanes. This proved to work quite well, although futher tuning and testing will be required for situations where the direction changes quickly. 
