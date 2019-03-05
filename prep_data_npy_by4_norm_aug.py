# -*- coding: utf-8 -*-
"""
Created on Tue Feb 3 2:22:37 2019

@author: Anoubhav
"""
# Importing Libraries
import os
import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


IMG_DIR = os.getcwd() + '\\images'
    
TRAIN_PATH = 'training.csv'
TEST_PATH  = 'test.csv'

# Reduced size by a fourth
IMG_SIZE = (160,120)
    
def horizontal_flip(img, box):
    # Flip the image horizontally
    img = img[:, ::-1]
    # Image center
    img_center = np.array(img.shape)[::-1]/2
    # Flip the bounding box
    box[0] += 2*(img_center[0]-box[0])
    box[1] += 2*(img_center[0]-box[1])
    t = abs(box[0]-box[1])
    box[0] -= abs(t)
    box[1] += abs(t)
    
    return img, box

def draw_rect(img, box):
    pt1, pt2 = (int(box[0]), int(box[2])) , (int(box[1]), int(box[3]))
            
    img = cv2.rectangle(img.copy(), pt1, pt2, color = [0]*3)
    return img
    
def create_dataset():
    
    training_data = pd.read_csv(TRAIN_PATH)
    testing_data = pd.read_csv(TEST_PATH)
    
    # training set
    X, y = [], []
    # Horizontally flipped images
    flip_X, flip_y = [], []
    # final testing set
    X_test = []
    # Ordered image names of the final testing set
    X_test_img_names = []

    # For identifying the set in which image belongs to, i.e. train, test or useless i.e. images neither in test.csv nor train.csv
    train_img_names = list(training_data['image_name'])
    test_img_names  = list(testing_data['image_name'])


    img_counter = 1  # For checking loop status
    error_counter, errors = 0, []  # For checking errors
    useless_counter = 0  # For checking useless images  


    for img_name in os.listdir(IMG_DIR):

        print(img_counter,' ---> ', img_name)
        img_counter += 1

        # Image path
        img_path = os.path.join(IMG_DIR, img_name)

        # Error: Assertion failed (ssize.width > 0 && ssize.height > 0) in cv::resize. This error occurs when file name has non-ASCII characters. These cases are captured in errors.
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)

            # Is image in training set
            if img_name in train_img_names:

                # Convert image into numpy array
                X.append(np.array(img)/255.)

                # label form: [x1, x2, y1, y2]. temp is a series containing that record
                temp = training_data[training_data['image_name'] == img_name]
                y.append([int(temp['x1']), int(temp['x2']), \
                    int(temp['y1']), int(temp['y2'])])
                
                # Horizontal flip augmentation for training images
                img_flip, box_flip = horizontal_flip(X[-1].copy(), (np.array(y[-1])/[4, 4, 4, 4]).copy())
                
                t = np.array(box_flip)*[4, 4, 4, 4]
                flip_X.append(img_flip)
                flip_y.append(t.astype(int))

            # Is image in test set
            elif img_name in test_img_names:
                X_test.append(np.array(img)/255.)
                X_test_img_names.append(img_name)
            # Image not in either train/test
            else:
                useless_counter += 1

        except:
            error_counter += 1
            errors.append((error_counter, img_name))
            continue
    
    #End of loop

    # Combine the normal images and horizontally flipped images
    X = np.concatenate((X, flip_X))
    y = np.concatenate((y, flip_y))
    
    # Save the numpy arrays locally. These will be used as input to DL models.
    np.save('X_by4_norm_aug.npy', X)
    np.save('y_aug.npy', y)
    np.save('X_test_by4_norm.npy', X_test)
    np.save('X_test_img_names.npy', X_test_img_names)

    # return and save image names of all errors
    print('No. of identified errors:', error_counter)
    #np.save('errors.npy', errors)

    # print number of useless images
    print('No. of useless images(neither train nor test):', useless_counter)
   
#     Visualize the horizontal flip augmentation
#    for k in range(10):
#        plt.figure(figsize=(8, 16))
#        plt.subplot(121)
#        plt.imshow(draw_rect(X[k], np.array(y[k])/[4, 4, 4, 4]), cmap = 'gray')
#        plt.subplot(122)
#        plt.imshow(draw_rect(flip_X[k], np.array(flip_y[k])/[4, 4, 4, 4]), cmap = 'gray')
#        plt.show()

 
def main():
    
    if os.path.exists('X_by4_norm_aug.npy') and os.path.exists('y_aug.npy') and os.path.exists('X_test_by4_norm.npy') and os.path.exists('X_test_img_names.npy'):
        print('Dataset partitions X_by4_norm_aug.npy, y_aug.npy, X_test_by4_norm.npy and X_test_img_names.npy have already been made. No need to run this file again.')
        sys.exit()
    
    input('-'*80+'\n\nPlease ensure that the correct path for images is given to IMG_DIR constant.\n\n'+'-'*80)
    input('-'*100+'\n\nPlease ensure that the correct path for training.csv and test.csv is given to TRAIN_PATH and TEST_PATH.\n\n'+'-'*100)
       
    create_dataset()
    
if __name__ == "__main__":
    main()



'''Identified 4 errors. As this number is small, these images will be ignored. And got 29972 useless images. Adding it all up 
29972(useless) + 14000(train) + 12815(test) + 4(error) = 56,791

Hence, program ran correctly as 56,791 images are present in IMG_DIR

The final size of training set: 4 GB.
The final size of test set: 3.66 GB'''









