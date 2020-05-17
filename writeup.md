# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./content_for_writeup/train.png "Train Visualization"
[image2]: ./content_for_writeup/valid.png "Valid Visualization"
[image3]: ./content_for_writeup/test.png "Test Visualization"
[image4]: ./content_for_writeup/Raw.png "Raw Image"
[image5]: ./content_for_writeup/Preprocessed.png "Preprocessed Image"

[image6]: ./test_data/20mph.jpg "Speed limit (20km/h)"
[image7]: ./test_data/80.jpg "Speed limit (80km/h)"
[image8]: ./test_data/deer.jpg "Wild animals crossing"
[image9]: ./test_data/dig.jpg "Road work"
[image10]: ./test_data/parenting.jpg "Children crossing"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated a summary statistics of the traffic
signs data set using python:

* The size of the training set is 34799
* The size of the testing set is 12630
* The size of the validation set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of classes in each set(training,testing,and validation)

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I preprocessed the data by converting images into grayscale and normalizing them. I could have augmented the dataset with scales and rotated images (since in real life signs could appear smaller or rotated but not mirrored) but I chose not to because I wanted to challenge myself and improve my accuracy through improvements to the model and not an extension of the dataset.

Here is a image pre and post preprocessing:
Both images are displayed in RGB:

![alt text][image4]
![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 BW image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Fully connected		| 400 inputs,  200 outputs 						|
| RELU					|												|
| Fully connected		| 200 inputs,  120 outputs 						|
| RELU					|												|
| Fully connected		| 200 inputs,  120 outputs 						|
| RELU					|												|
| Fully connected		| 120 inputs,  number of classes outputs 		|
| Softmax 				|												|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used Cross Entropy Loss since it generally performs well for classification problems and I used the Adam Optimizer because it is usually the default I go to.

The model is trained with a 0.5 dropout probability for all fully connected layers

I used a learning rate of 0.001 which is the same as LeNet and trained for 70 epochs with a 128 batchsize (same as the LeNet implementation).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.963 
* test set accuracy of 0.937
I calculated this after my implementation (code cell 12-14)

I tried an iterative approach

I started with LeNet and my current preprocessing pipeline since I had previously written it in TensorFlow.
It worked well but the accuracy hovered in the mid 80s. 
I then added dropout knowing that the robustness of the classification portion of my neural net would improve. The accuracy hit the low 90s.
I then added another fully connected layer knowing that the needed features were already being extracted (the convolutional layers did not need to be altered) but that the classification portion could be improved.
I then had to change the number of epochs because, with more parameters to learn, validation accuracy was not plateauing in the initial epoch rate of 27. I changed it to 50 but was underfitting. I then updated it to 70 and approached overfitting (my training accuracy is higher than validation which is higher than the testing set)
I then hit a test set accuracy higher than the minimum validation set accuracy so I stopped iterating.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The images for 80mph speed limit, 20mph speed limit, and the wild animal crossing signs are distorted by the angle of view and the image for the road work sign has visible weather both of which would make them hard to classify without a linear transformation.
All of them are relatively clear and have good lighting but these were all traits we had trained to adapt to.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing		| Bicycles crossing   							| 
| Wild animals crossing	| No passing for vehicles over 3.5 metric tons  |
| Road work				| Road work										|
| Speed limit (80km/h) 	| Speed limit (80km/h)	 						|
| Speed limit (20km/h)	| Roundabout mandatory 							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares unfavorably to the accuracy on the test set of 94% and is most likely caused by the variations in angles and an augmentation of the dataset would likely be beneficial to increase the web dataset accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively is unsure what the sign is predicting the Bicycles crossing with less than 50% certainty. The second highest prediction, Children crossing, is the correct label.

#### Children crossing
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 47.6962476969 %		| Bicycles crossing 							| 
| 13.9129579067 %		| Children crossing								|
| 13.2450133562 %		| Dangerous curve to the right					|
| 4.9482896924 %		| Road work 					 				|
| 4.68202866614 %	    | Beware of ice/snow  							|

For the second image, the model is relatively sure the sign is predicting No passing for vehicles over 3.5 metric tons. This is not the correct label.

####  Wild animals crossing
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 91.9260323048 %		| No passing for vehicles over 3.5 metric tons  | 
| 3.69715616107 %		| Slippery road 								|
| 2.5759935379 %		| Wild animals crossing							|
| 1.13011868671 %		| Dangerous curve to the left 		 			|
| 0.436750799417 %	    | No passing 									|

For the third image, the model is relatively sure the sign is predicting Road work and it is right. This is the correct label.

####  Road work
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.8037278652 %		| Road work  									| 
| 0.114766240586 %		| Bicycles crossing 							|
| 0.0166428144439 %		| Bumpy road									|
| 0.0161749179824 %		| Priority road 					 			|
| 0.016067159595 %	    | Dangerous curve to the right 					|

For the fourth image, the model is relatively sure the sign is predicting Speed limit (80km/h) and it is right. This is the correct label.

####  Speed limit (80km/h)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.973398447 %		| Speed limit (80km/h)							| 
| 0.0108596075734 %		| Speed limit (60km/h) 							|
| 0.0083843820903 %		| Speed limit (50km/h) 							|
| 0.0051037502999 %%	| Speed limit (100km/h) 						|
| 0.00224645245908 %	| Speed limit (30km/h)  						|

For the final image, the model is moderately sure the sign is predicting Roundabout mandatory it is wrong. The correct label is Speed limit (20km/h) and it does recognize that it might be a speed limit sign (the next three are speed limits) but fails to recognize the 20 in the sign.

####  Speed limit (20km/h)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 76.6255617142 %		| Roundabout mandatory							| 
| 10.0643016398 %		| Speed limit (30km/h) 							|
| 5.26306442916 %		| Speed limit (100km/h) 						|
| 2.35923118889 %		| Speed limit (70km/h)  						|
| 2.17697042972 %		| Priority road  								|

This could be used to argue that more feature extraction is needed.
