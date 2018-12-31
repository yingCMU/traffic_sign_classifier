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

[image1]: ./outputs/train_data_dist.png "Train Distribution"
[image2]: ./outputs/original.png "Original"
[image3]: ./outputs/gray_normalized.png "Normalized Gray"
[image4]: ./outputs/test_5_images.png "Test 5 Images"
[image5]: ./outputs/visualization_layers.png "Visualization first conv layer"


---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because experiments in the paper reported higher accuracy in gray scale. And I also normalized the image because it converges much faster with feature scaling than without it.

Here is an example of a traffic sign image before and after grayscaling and normalization

![alt text][image2]
![alt text][image3]

The paper said augumenting the image data led to better accuracy but I didn't implement this since I got 95% acuracy without this step. I will try this out later.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flattern       		| outputs 400 									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Dropout				| keep 0.5										|
| Fully connected		| outputs 84									|
| RELU					|												|
| Dropout				| keep 0.5										|
| Fully connected		| outputs 43									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an epoch size of 30, batch size of 128, Adam optimization algorithm, learning rate of 0.0009 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.955
* test set accuracy of 0.933


If a well known architecture was chosen:
* What architecture was chosen? Why did you believe it would be relevant to the traffic sign application?
I chose the LeNet architecture from the lecture because it is classifying letters, similar to traffic signal classification. Initially I modified network to fit 3 color channels. The initial architecture only gives 0.88 - 0.89 accuracy on validation set, and similay accuracy on test set. which is far from traffic signal classifier accuracy requirements.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Both test and train accuracy are low indicating under fitting. I converted data to grayscale because it was reported in the paper that grayscale has better accuracy. Surprisingly turing into gray scale doesn't change much.
 I increased the epoch from 10 to 30, learning rate reduced to 0.0009 to get validation accuracy of 0.90 - 0.93, test accuracy of 0.899, which indicates onverfitting. 
 So I added dropout of keep rate 0.5. This makes validation accuracy increase to 0.955, test accuracy increase to 0.933.
 I didn't have time to add random noise to increase training set though.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The 3rd image might be difficult to classify because brightness is very low. The 2nd image seems to have been blured.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image										|Prediction								| 
|:------------------------------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)| Speed limit (30km/h)| 
| No passing for vehicles over 3.5 metric tons	|No passing for vehicles over 3.5 metric tons|
| Speed limit (80km/h)			| Speed limit (80km/h)			|
| Right-of-way at the next intersection| Right-of-way at the next intersection|
| Turn right ahead			| Turn right ahead			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.933

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the 3rd image, the model is relatively sure that this is a 'Speed limit (80km/h)' sign (probability of 0.93)The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.93         			| Speed limit (80km/h)  		 		 		| 
| 0.02    				| Speed limit (120km/h) 						|
| 0.02					| Speed limit (60km/h)							|
| 0.02	      			| Speed limit (50km/h)					 		|
| 0.01				    | Speed limit (100km/h)    						|

 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I visualize the 3rd image's first convolution network layer, curve are used to make classifications
![alt text][image5] 
