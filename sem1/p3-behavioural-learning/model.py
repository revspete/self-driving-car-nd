import csv
from PIL import Image
import cv2
import numpy as np
import h5py
import os
from random import shuffle
import sklearn

# Read and store lines from driving log csv file
samples = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	# if we added headers to row 1 we better skip this line
	#iterlines = iter(reader)
	#next(iterlines)
	for line in reader:
		samples.append(line)

# Take a 20% validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

images = []
measurements = []
def process_image(image):
	# do some pre processing on the image
	# TODO: Continue experimenting with colour, brightness adjustments
	return image

# TODO: more testing with ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
# https://keras.io/preprocessing/image/
#train_datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=0,
#    width_shift_range=0.0,
#    height_shift_range=0.0,
#    horizontal_flip=True)
	
# Info: https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
def generator(samples, batch_size=32):
	# Create empty arrays to contain batch of features and labels
	num_samples = len(samples)
	
	while True:
		shuffle(samples) 
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			batch_features = [] #np.zeros((batch_size, 160, 320, 3))
			batch_labels = [] #np.zeros((batch_size, 1))
	
			for batch_sample in batch_samples:
				path = "data/IMG/"
				img_center 	= process_image(np.asarray(Image.open(path+os.path.basename(batch_sample[0]))))
				img_left 	= process_image(np.asarray(Image.open(path+os.path.basename(batch_sample[1]))))
				img_right 	= process_image(np.asarray(Image.open(path+os.path.basename(batch_sample[2]))))	
				
				#We now want to create adjusted steering measurement for the side camera images
				steering_center = float(batch_sample[3]) # steering measurement for centre image
				correction = 0.1 # steering offset for left and right images, tune this parameter
				steering_left = steering_center + correction
				steering_right = steering_center - correction
                
				# TODO: Add throttle information
				batch_features.extend([img_center, img_left, img_right, cv2.flip(img_center,1), cv2.flip(img_left,1), cv2.flip(img_right,1)])
				batch_labels.extend([steering_center, steering_left, steering_right, steering_center*-1.0, steering_left*-1.0, steering_right*-1.0])
			
			X_train = np.array(batch_features)
			# Do some image processing on the data
			#train_datagen.fit(X_train)
			y_train = np.array(batch_labels)
			yield sklearn.utils.shuffle(X_train, y_train) # once we've got our processed batch send them off

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Imports to  build the model Architecture
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers.core import Dropout
from keras.layers.noise import GaussianDropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D

# In the architecture we add a crop layer
crop_top = 65
crop_bottom = 22
# The input image dimensions
input_height = 160
input_width = 320
new_height = input_height - crop_bottom - crop_top

# Build the model architecture
# Based on http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()
model.add(Cropping2D(cropping=((crop_top,crop_bottom),(0,0)), input_shape=(input_height,input_width, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(new_height,input_width,3)))
model.add(Conv2D(24,kernel_size=5,strides=(2, 2),activation='relu'))
model.add(Conv2D(36,kernel_size=5,strides=(2, 2),activation='relu'))
model.add(Conv2D(48,kernel_size=5,strides=(2, 2),activation='relu'))
model.add(Conv2D(64,kernel_size=3,strides=(1, 1),activation='relu'))
model.add(Conv2D(64,kernel_size=3,strides=(1, 1),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print("model summary: ", model.summary())

batch_size = 100
# Info: https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
history_object = model.fit_generator(
	train_generator, steps_per_epoch=len(train_samples)/batch_size, 
	validation_data = validation_generator, validation_steps=len(validation_samples)/batch_size,
	epochs=7, verbose=1)
	
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

