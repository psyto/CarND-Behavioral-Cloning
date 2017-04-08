**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on Nvidia model.
I added dropout function inspired by Comma.ai model to it.

####2. Attempts to reduce overfitting in the model

As Keras was upgraded to version 2, my autonomous driving became not working.
I could make the code compatible with Keras 2 easily, but the training speed became really slow, nearly 10 times slower than before.
Training with 3 epochs took around 1 hour with Keras 1, but it took almost 1 day with Keras 2.

I tried AWS GPU with Udacity AIM.
I surprised that training with 3 epochs took only 5 min to complete on AWS GPU.

As AWS GPU enabled me to try many things in short time, I increased the number of epochs to 10 at first (model.py line 134).
To avoid overfitting, I added dropout layer among first 3 Dense layers (model.py line 113 and 115).

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 132).

####4. Appropriate training data

I used the data provided by Udacity.

Initially, I collected data myself by driving a car around a track in Udacity's simulator.
But, due to low quality of my data, my autonomous driving was always like a drunken driving.

By using Udacity data, my autonomous driving became almost fine.
I decided going with Udacity data, but my autonomous driving could not recovery from the edge lanes.

I consulted my mentor Sumit and he suggested using left and right images with correction +- 0.2.
By using left and right images (model.py lines 71-80), my autonomous driving became able to recovery from the edge lanes.

My autonomous driving was still off the load after the bridge.
I recorded the vehicle recovering from the left side and right sides of the road back to center, but it could not resolve the issue.

I consulted my mentor Sumit again and he introduced the excellent blog post by Vivek Yadav https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9.
By using the data augment functions (model.py lines 25-58), my autonomous driving became keeping on the road for one lap, but it was only once per 10 tries.

As the results were not constant, I tried other augmentation, but it did not help much.
I stopped adding data augmentation and started updating the model.

###Model Architecture and Training Strategy

####1. Solution Design Approach

Originally, I trained my model on my local machine, which does not have GPU.
I used the data provided by Udacity, but, with full data, one training took nearly 1 hour.
So, to check if my code working, I created sample data by using 100 data of Udacity data.
After checking my code, I trained my model with full data.

After Keras was upgraded to version 2, my model became not working.
I corrected the code to make it work with Keras 2.
But, the training speed became really slow.
So, I decided using AWS GPU with udacity-carnd AIM.

With AWS GPU, I could become able to try many model variations,

I tried many data augmentation and found random brightness and shadow effects are effective for the lanes without lines (model.py lines 25-58).
I tried other data augmentation, but they made my autonomous driving worse (looks like a drunken driving).

With AWS GPU, I could try larger epochs.
Validation loss increased a bit at epoch 5, but it decreased again until epoch 10.
So, I decided going with epoch 10 (model.py line 134).

In addition, I got surprised with the strength of generator function.
When I trained my model on my local machine, using generator function is required to avoid memory error.
Even with AWS GPU, generator made the training faster, I kept using it in my final model (model.py lines 60-98).

####2. Final Model Architecture

I tried many variations of Nvidia model.
My final model architecture is based on Nvidia model with dropout function inspired by Comma.ai model.
My final model consists of the below layers.

Lambda layer to normalize the data (model.py line 104)
Cropping layer to crop useless parts (70 pixel from top and 25 pixel from bottom) of the images (model.py line 105)
Convolution layer with 24 filters, a 5x5 kernel, a 2x2 subsample, and Relu activation to introduce nonlinearity (model.py line 106)
Convolution layer with 36 filters, a 5x5 kernel, a 2x2 subsample, and Relu activation (model.py line 107)
Convolution layer with 48 filters, a 5x5 kernel, a 2x2 subsample, and Relu activation (model.py line 108)
Convolution layer with 64 filters, a 3x3 kernel, and Relu activation (model.py line 109)
Convolution layer with 64 filters, a 3x3 kernel, and Relu activation (model.py line 110)
Flatten layer to make the results into 1 dimention (model.py 111)
Dense layer to categorize the results into 100 (model.py 112)
Dropout layer to avoid overfitting (model.py line 113)
Dense layer to categorize the results into 50 (model.py line 114)
Dropout layer to avoid overfitting (model.py line 115)
Dense layer to categorize the results into 10 (model.py line 116)
Dense layer to categorize the results into 1 (model.py line 117)

####3. Creation of the Training Set & Training Process

I split data into 80% training set and 20% validation set by using sklearn train_test_split function (model.py line 124).

I shuffled the training set by using sklearn.utils.shuffle function (model.py line 64).

In addition to center images, I appended left and right images to the training set by using for loop (model.py lines 71-80).

I applied random brightness and shadow effects to the training set by using Vivek's data augment functions (model.py lines 89-90).

I flipped center images and appended them to the training set (model.py lines 91-94).
