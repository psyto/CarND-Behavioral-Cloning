# Behavioral Cloning

# Import required packages
import csv
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read driving_log.csv provided by Udacity.
# driving_log.csv contains 8036 rows.
def read_data():
    with open('./data/driving_log.csv') as csvfile:
#    with open('./data/driving_log_sample.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

# Augment images by applying random brightness
# Using the augment function explained by Vivek Yadav https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def add_random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

# Augment images by applying random shawdom
# Using the augment function explained by Vivek Yadav https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

# Use Keras generator to put only required data on memory.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Use center, left, and right images.
                for i in range(3):
                    name = './data/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)

                correction = 0.2
                angle = float(batch_sample[3])
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)

            # Augment the data set
            augmented_images = []
            augmented_angles = []

            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                image1 = add_random_brightness(image)
                image1 = add_random_shadow(image1)
                flipped_image = cv2.flip(image, 1)
                flipped_angle = angle * -1.0
                augmented_images.append(flipped_image)
                augmented_angles.append(flipped_angle)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Use the model inspired by Nvidia and Comma.ai models
def get_model():
    # Construction
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

if __name__ ==  '__main__':

    samples = []
    read_data()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # Execute
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = get_model()

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
        validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=10)

    model.save('model.h5')
