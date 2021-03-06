# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:28:44 2019

@author: Anoubhav
"""

# Library imports
import numpy as np
np.random.seed(42)
import os
import sys
import shutil
import pandas as pd
from imagenet_architectures.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

TEST_PATH = 'test.csv'

NUM_CLASSES = 4

# Creates folder for storing model checkpoints
newpath = os.path.join(os.getcwd(), 'InceptionResNetv2_checkpoints')
if not os.path.exists(newpath):
    os.makedirs(newpath)
        
# Load training dataset created using dataset_prep_csv_files.py
def load_dataset():
    if os.path.exists('X_by4_norm_aug.npy') and os.path.exists('y_aug.npy'):
        X = np.load('X_by4_norm_aug.npy')
        y = np.load('y_aug.npy')
        
        # Normalize target label to [0, 1] 
        y = y/[640, 640, 480, 480]
        print('Successfully loaded X and y')
    
        # Train_val split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
        
        # Stack 3 times to satisfy model input shape
        X_train = np.stack([X_train]*3, axis= -1)
        X_test = np.stack([X_test]*3, axis= -1)
        
        # Remove memory of redundant variables
        del X, y
        
        return X_train, X_test, y_train, y_test
    else:
        print('-'*50+'\n')
        print('Please execute dataset_prep_csv_files.py first!')
        print('\n'+'-'*50)
        sys.exit()

def create_model():
    # Image shape
    image_input = Input(shape=(120, 160, 3))

    # Set weights to none. Remove dense layers of model
    model = InceptionResNetV2(input_tensor=image_input, include_top=False, weights=None)

    # Add custom fully connected layers
    last_layer = model.get_layer('conv_7b_ac').output
    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    out = Dense(NUM_CLASSES, activation='linear', name='output')(x)
    
    return Model(image_input, out)

def get_predictions():
    model = create_model()
    
    # Path of directory containing all model checkpoints
    path = os.path.join(os.getcwd(), 'InceptionResNetv2_checkpoints')
    
    # Finds the checkpoint with lowest val_loss. This checkpoint is loaded to model.
    min_val_loss = 100000   # arbitrary large number
    min_filename = ''
    for file in os.listdir(path):
        temp = file.split('-')[1][:-5]
        if float(temp) < min_val_loss:
            min_val_loss = float(temp)
            min_filename = file
            
    # Load model weights        
    model.load_weights(os.path.join(path, min_filename))
    print('Model loaded')
    
    # Load Final test
    final_test = np.load('X_test_by4_norm.npy')
    final_test = np.stack([final_test]*3, axis= -1)
    preds = model.predict(final_test)
    
    # Bringing back to scale
    preds = preds*[640, 640, 480, 480]   
    preds = preds.astype(int)
    
    test = pd.read_csv(TEST_PATH, index_col='image_name')
    X_test_names = np.load('X_test_img_names.npy')

    for i in range(len(X_test_names)):
        temp = preds[i]
        test.loc[X_test_names[i], 'x1'] = temp[0]
        test.loc[X_test_names[i], 'x2'] = temp[1]
        test.loc[X_test_names[i], 'y1'] = temp[2]
        test.loc[X_test_names[i], 'y2'] = temp[3]
    
    test.x1 = test.x1.astype('int')
    test.x2 = test.x2.astype('int')
    test.y1 = test.y1.astype('int')
    test.y2 = test.y2.astype('int')
    
    test.to_csv('predictions_InceptionResNetV2.csv', encoding='utf-8', index=True)
    
    
def main():
    model = create_model()
    X_train, X_test, y_train, y_test = load_dataset()

    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    checkpoint = ModelCheckpoint("./InceptionResNetv2_checkpoints/model.{epoch:02d}-{val_loss:.4f}.hdf5", monitor="val_loss", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="min", period=1)

    model.fit(X_train, y_train, batch_size=32, epochs=72, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint])
    
    del X_test, X_train, y_test, y_train
    
    get_predictions()

        ## Delete the InceptionResNetv2_checkpoints directory
    # try:
    #     shutil.rmtree('InceptionResNetv2_checkpoints')
    # except OSError as e:
    #     print ("Error: %s - %s." % (e.filename, e.strerror))
    
if __name__ == "__main__":
    main()