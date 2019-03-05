# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:34:08 2019

@author: Anoubhav
"""
# Importing libraries
import numpy as np
np.random.seed(42) # for reproducibility
import pandas as pd
from sklearn.utils import shuffle
import csv
import math
import glob
import cv2
from PIL import Image
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import Sequence
from tensorflow.keras.backend import epsilon

TRAIN_PATH = 'training.csv'
TEST_PATH = 'test.csv'
ALPHA = 1.4

IMAGE_SIZE = (160, 120)

IMAGES = "test_images/*png"
# The model gets early stopped at 175 epochs.

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 50

TRAIN_CSV = "train_3.csv"
VALIDATION_CSV = "val_3.csv"

NUM_CLASSES = 4

# Creates folder for storing model checkpoints
newpath = os.path.join(os.getcwd(), 'Mobilenet_checkpoints')
if not os.path.exists(newpath):
    os.makedirs(newpath)

def create_train_val_csv(seed = 42, train_val_split = 0.2):
    df = pd.read_csv(TRAIN_PATH)
    df['height'] = 480
    df['width']  = 640
    df = df[['image_name', 'height', 'width', 'x1', 'y1', 'x2', 'y2']]
    df['image_name'] = 'train_images/' + df['image_name']
    
    df = df.set_index('image_name')

    df = shuffle(df, random_state = seed)

    row_no = int(df.shape[0]*train_val_split)

    df_val   = df.iloc[ : row_no, :]
    df_train = df.iloc[row_no : , :]

    df_train.to_csv('train_3.csv', index = True, header = False)
    df_val.to_csv('val_3.csv', index = True, header = False)
    
    del df, df_val, df_train
    
def get_predictions():
    model = create_model()
    
    # Path of directory containing all model checkpoints
    path = os.path.join(os.getcwd(), 'MobileNet_checkpoints')


#   Filename format: model-1-{epoch:02d}-{val_iou:.2f}-.h5

    # Finds the checkpoint with lowest val_loss. This checkpoint is loaded to model.
    max_iou = 0   # arbitrary large number
    min_filename = ''
    for file in os.listdir(path):
        if file.split('-')[1] == '3':
            temp = file.split('-')[3]
            if float(temp) > max_iou:
                max_iou = float(temp)
                min_filename = file
            
    # Load model weights        
    model.load_weights(os.path.join(path, min_filename))
    print('Model loaded')
    
    test = pd.read_csv(TEST_PATH, index_col='image_name')
    
    count = 0
    for filename in glob.glob(IMAGES):
        if count%1000==0: print('Number of images predicted:', count)
        count += 1
        unscaled = cv2.imread(filename)
        # image_height, image_width, _ = unscaled.shape
        image_height, image_width = 480, 640
        try:
            image = cv2.resize(unscaled, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
            feat_scaled = preprocess_input(np.array(image, dtype=np.float32))
    
            region = model.predict(x=np.array([feat_scaled]))[0]
    
            x1 = int(region[0] * image_width / IMAGE_SIZE[0])
            y1 = int(region[1] * image_height / IMAGE_SIZE[1])
    
            x2 = int((region[0] + region[2]) * image_width / IMAGE_SIZE[0])
            y2 = int((region[1] + region[3]) * image_height / IMAGE_SIZE[1])
            
#            filename contains test_images/ which is not there in test.csv
            test.loc[filename[12:], 'x1'] = x1
            test.loc[filename[12:], 'x2'] = x2
            test.loc[filename[12:], 'y1'] = y1
            test.loc[filename[12:], 'y2'] = y2
        except:
            print(count, filename)
    
    test.to_csv('predictions_MobileNet_model3.csv', encoding='utf-8', index=True)
        
class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.coords = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)
                path, image_height, image_width, x0, y0, x1, y1 = row
                self.coords[index, 0] = x0 * IMAGE_SIZE[0] / image_width
                self.coords[index, 1] = y0 * IMAGE_SIZE[1] / image_height
                self.coords[index, 2] = (x1 - x0) * IMAGE_SIZE[0] / image_width
                self.coords[index, 3] = (y1 - y0) * IMAGE_SIZE[1] / image_height 
                self.paths.append(path)

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), 160, 120, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((120, 160))
            img = img.convert('RGB')
            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return batch_images, batch_coords

    
class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0
        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)
            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)

            diff_width = np.minimum(gt[:,0] + gt[:,2], pred[:,0] + pred[:,2]) - np.maximum(gt[:,0], pred[:,0])
            diff_height = np.minimum(gt[:,1] + gt[:,3], pred[:,1] + pred[:,3]) - np.maximum(gt[:,1], pred[:,1])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:,2] * gt[:,3]
            area_pred = pred[:,2] * pred[:,3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse

        print(" - val_iou: {} - val_mse: {}".format(iou, mse))

def create_model(trainable=True):
    image_input = Input(shape=(160, 120, 3))
    model = MobileNetV2(input_tensor=image_input, include_top=False, alpha=ALPHA, weights = None)
    last_layer = model.layers[-1].output
    
    x = GlobalAveragePooling2D()(last_layer)
    x = layers.GaussianDropout(0.3)(x)
    x = Dense(512, activation=layers.ELU(alpha=1.0), name='fc1')(x)
    x = layers.GaussianDropout(0.1)(x)
    x = Dense(64, activation=layers.ELU(alpha=1.0), name='fc2')(x)
    x = layers.GaussianDropout(0.05)(x)
    x = Dense(NUM_CLASSES, activation='linear', name='output')(x)
    
    return Model(inputs=model.input, outputs=x)

def main():
    create_train_val_csv(seed = 24, train_val_split = 0.2)
    
    model = create_model()
    model.summary()

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=[])  
    
    checkpoint = ModelCheckpoint("./Mobilenet_checkpoints/model-3-{epoch:02d}-{val_iou:.2f}-.h5", monitor="val_iou", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max", period=1)
        
        
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")

    model.summary()
    model.fit_generator(generator=train_datagen,
                        epochs=EPOCHS,
                        callbacks=[validation_datagen, checkpoint, reduce_lr, stop],
                        shuffle=True,
                        verbose=1)
    get_predictions()

if __name__ == "__main__":
    main()