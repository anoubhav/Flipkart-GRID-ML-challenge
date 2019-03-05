We used **approach 2** which was developed and explained in round 2 source file. Key features of this approach:
- A *data generator* was built with keras. A data generator can generate the dataset on multiple cores in real time and feed it right away to the deep learning model.
- Also, the *IOU metric was implemented*. The model checkpoints would now be created based on maximum validation IOU. We still use the mse loss function.
- The functionality for resizing images is now directly incorporated into the data generator. The images can be resized freely without having to run any dataset preparation script multiple times.

## Dataset Preparation
- In approach 1, the images were screened and then converted to numpy arrays. Now, instead of converting them to numpy arrays, the screened images are separated into 2 different directories. These are train_images and test_images. This is done by checking if the image name is present in the training.csv or test.csv.

- The path to these two directories will then be fed to the data generator class for generating training/validation data during training and testing data while making predictions. The prep_data_approach2.py executes the above process.

- The preprocessing is now done using preprocess_input from keras.applications.[ModelArchitecture]. In our case, it will be mobilenet_v2. We don't normalize the pixels in the dataset preparation this time. The preprocess_input handles these transformations. This function is present in the model file itself.

- In the train_images directory, some of the images will be used for validation. The separation of training images into train and validation is done in the model file itself. The function used is create_train_val_csv() .It takes two arguments. These are:

    - Random seed: This is used to initialize a pseudorandom number generator. Here, we use it to randomly shuffle our training images.
    - Train_val_split: it signifies what fraction of the training images will be used for validation.


## Network Architectures
- We only built a single **Xception model** due to time constraints. The Xception model was chosen as it gave great results in round 2 where we had used it extensively. It achieved a Public LB score of *0.896*. This is present in the file *r3_Xception_model.py*


