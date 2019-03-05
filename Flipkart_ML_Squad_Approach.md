# Approach
This markdown file goes into the few details behind how we approached the problem from scratch. 

**Note:** To just test and run the source files check **Quick_start.md** instead.

The approach is broadly divided into 2 basic pipelines. These are:
1. Loading the dataset entirely at once.
2. Loading the dataset in batches through custom data generators.

We will look at these 2 approaches sequentially.

We have predominantly used the keras API for building our models and pipeline. 
The system specifications used by us: **NVIDIA 1050** card with **16 GB RAM** on board. 

We estimate the reproducibility of our results to be **atmost 2% lower** than the Public Leaderboard score. This is due to two reasons:
- More number of models were stacked for submission of public leaderboard scores  compared to the ones mentioned in this final source file. This was done because we were unable to package them into single code file making it extremely convoluted.
  
- We are using a GPU which employs extreme **multithreading** and the inherent randomness is tough to get rid of when using GPU computing. 

- Stacking of models is being done by taking simple average in this source file and not weighted average as done during the contest. Reason for this is mentioned later.
   
## Approach 1: Dataset Preparation 
- The downloaded image dataset contained 56,791 images and the total size was 13.3 GB. All images had dimensions of **640x480**.
  
- It was observed that train.csv and test.csv had 14,000 and 12,815 images respectively. 
- Hence, we ran a script to screen only the relevant images and create numpy arrays X, y and X_test. 
- X and y numpy arrays correspond to the images and labels mentioned in train.csv. X_test corresponds to the images in test.csv
  
- We first loaded the X.npy at once into the workflow which had a size of 4 GB and this array had to be stacked 3 times to be fed as input to the models. 
- Hence 12 GB out of 16 GB RAM was used in just loading the train data. As 20+ % of RAM was already occupied by the background process, we were operating at 95+ % memory. Which caused ResourceExhaustionErrors.
- To avoid these problems and speed up the initial phase of making baseline submission, the script to screen the image dataset was modified. 
- The **prep_data_npy_by4_norm_aug.py** loops through all the images and divides them into X, y and X_test. 
    - It **resizes** the images into the shape of **(120, 160)** which is a fourth of the initial image size. 
    - The images were **converted to grayscale** based on the signal to noise argument that color information doesn't help us identify important edges or other features. 
    - For preprocessing we **normalized** the pixel values and brought them from [0, 255] to [0, 1]. 
    - We avoided standardizing the data (zero mean and unit variance) due to poorer results compared t based on articles.
    - The python script performs **horizontal flip augmentation** as well. However, this feature was added much later.
- After running this the size of X was reduced from 4 GB to 2 GB making it more workable on our machine specs. 
- The **target label y** was modified in the following ways: 
    1. By **normalizing** its range to [0, 1]. This is done by y/[640, 640, 480, 480].
    2. To add **diversity in some models**, y was modified to y/[4, 4, 4, 4] . As the image size was reduced by a fourth, the bounding box coordinates will also decrease by the same.
    3. Using different monotonic target functions such as: **yt = log(1+y)** and **yt = 1/y**. To scale the predictions back to normal, the inverse is taken which is **exp(yt)-1** and **1/yt** respectively. 
    4. **yt = 1/y** was not used due to some errors before deadline
   
- While making predictions the target label y is brought to the same scale as the challenge expects absolute values of the bounding box coordinates for an image of size 640x480.
## Approach 1: Network Architectures
- After dataset preparation, the object localization task was tested using the widely used **imagenet architectures**.
  
- We have **not** used the pretrained Imagenet weights as stated by the rules. All models have been **trained from scratch**.
  
- For all the architectures the following things were common: 
    - The **output activation** was **linear**

    -  We did not use the Dense layers of these architectures (**include_top = False**). Most of the hyperparameter tuning was around the custom layers added to these imagenet **feature extractors**.  
    -  **Loss function: mean-squared-error**. We also tested mean absolute error loss. However, mse (L2 loss) performed better when it came to obtaining a higher IOU score. The main benefit of mae (L1 loss) is that it is more robust to outliers. However, the training dataset was quite clean and the distribution of images was far from erratic.

- Efforts were made into implementing the **Weighted Hausdorff Distance** for loss function as the difference in the accuracy obtained in minimizing the mse loss and the **IOU score** (Intersection over Union) was substantially high around 7-10%.

- We first started with **ResNet50 and VGGNet**. Both these architectures gave **suboptimal results**. We reached an LB (Leaderboard) score of **0.78** using ResNet50 after multiple iterations and tweaks. That model had given a **val_acc of 89%** .
-  Due to this large difference in the val_acc and IOU score, we inferred that the data needs to be **augmented** as these architectures had millions of parameters but only 12K images for training (after train_test split).

- Due to memory constraints and also our approach of loading entire dataset at once, we could only perform **horizontal flip augmentation**. This is done using the **data_aug_hor_flip.py** script.
  
- The major breakthrough in IOU score happened after using the more recent imagenet architectures. The augmented data was fed to the Xception architecture. **Xception or Extreme Inception** developed by Google uses a modified depthwise separable convolution and is even better than the Inception-v3. Finetuning this model had given us an LB scores from **0.89 - 0.906**. We will be using the best Xception architecture amongst these for stacking. The **Xception_model.py** file contains this model. Here, the labels y were scaled to a fourth of the original labels.

- As the Xception model gave Xceptional(pun intended) results. The new strategy was to develop more models having a similar IOU score of 0.89-0.9 and then taking the **weighted average of their predictions**. The weight assigned to the model would be based on its public LB IOU score. This strategy had given us an LB IOU score of **0.91535** by stacking many relatively different models. 

- A way of effective stacking especially in case of regression is done by different **monotonic target transformations**. Here, based on the success of the Xception model, we performed the following target transformations on y: **yt = log(1+y)** and **yt = 1/y**. This is done in the python files *Xception_model_log_transform.py* and *Xception_model_reciprocal_transform.py* respectively. We also changed the seeds in these files for diversity in data.

**NOTE: *Xception_model_reciprocal_transform.py* not included because of errors before deadline.**


- For our source file, the fourth model we are going to stack is an **InceptionResNetV2**. Very minimal hyperparameter tuning was done for this as the model was huge, having about **55 million** parameters and was thus not feasible to run multiple times for our machine specs. However, without , we obtained an LB IOU score of 0.905256. The **InceptionResNetV2_model.py** file contains this model. Here, the labels y were scaled to [0, 1] range.

- Loading models from keras.applications might give errors such as **AttributeError: 'InputLayer' object has no attribute 'outbound_nodes'** due to different versions of keras and lag in updates from tf.keras for this newer API. Hence, to be safe the architectures are saved locally in the **directory imagenet architectures** , instead of loading it from keras.applications.
  

## Approach 2
After achieving LB IOU score of 0.9 using approach 1, the focus was shifted from getting marginally higher IOU to mainly addressing the shortcomings of approach 1. Thus, building a **flexible data pipeline** for more complex data was the primary goal.

The **major shortcomings** of approach 1 were:

- The training dataset was loaded at once in approach 1. This puts a cap on the size of the training data that our model can use to learn. Hence, even data augmentation capabilities became limited. For the system specifications used, we had almost reached the cap. 
  
- **IOU metric was not used** in creating our model checkpoints(i.e. saving model weights after an epoch, if a certain condition is met). The checkpoints were created based on minimum validation loss. Due to this, there was a discrepancy between what was the expected IOU and the real IOU score given by submission solver. The mse loss has issues with stability. **A lower mse loss does not always imply a lower IOU score**.

- Any form of pre-processing(such as denoising, resizing images, segmentation etc.) was not possible when the dataset was loaded at once due to **MemoryError** and **ResourceExhaustedError**.

- Even just for resizing images, the script **prep_data_npy_by4_norm_aug.py** had to be executed from scratch. Creating large images would once again lead to above-mentioned errors. In approach 1, a compromise was made and the image size was reduced by 4 in order for the dataset to fit into memory. Increasing image size helps in increasing the accuracy of the object localization task. Clear evidence for this is observed in the case of MobileNets where the Top-1 and Top-5 accuracy increase with the increase in input image size (Accuracy order: 224> 192 >160).

To solve these shortcomings the following components were modified/added:
- A **data generator was built** with keras. A data generator can generate the dataset on multiple cores in real time and feed it right away to the deep learning model. 
  
- Also, the **IOU metric was implemented**. The model checkpoints would now be created based on maximum validation IOU. We still use the mse loss function.
  
- The functionality for resizing images is now directly incorporated into the data generator. The images can be resized freely without having to run any dataset preparation script multiple times. 
  
- Other different pre-processing techniques and data augmentation techniques can now be added to the data generator class without much issue of leftover memory space. The batch size will now primarily dictate how much memory is being used in the process.


## Dataset Preparation
- In approach 1, the images were screened and then converted to numpy arrays. Now, instead of converting them to numpy arrays, the screened images are **separated into 2 different directories**. These are **train_images** and **test_images**. This is done by checking if the image name is present in the training.csv or test.csv.
  
- The path to these two directories will then be fed to the data generator class for generating training/validation data during training and testing data while making predictions. The **prep_data_approach2.py** executes the above process.

- The preprocessing is now done using **preprocess_input** from **keras.applications.[ModelArchitecture]**. In our case, it will be mobilenet_v2. We don't normalize the pixels in the dataset preparation this time. The preprocess_input handles these transformations. This function is present in the model file itself.
  
- In the train_images directory, some of the images will be used for validation. The separation of training images into train and validation is done in the model file itself. The function used is *create_train_val_csv()* .It takes two arguments. These are:
  - *Random seed*: This is used to initialize a pseudorandom number generator. Here, we use it to randomly shuffle our training images.
  - *Train_val_split*: it signifies what fraction  of the training images will be used for validation.

## Network Architectures
- Due to time constraints, the second approach was mainly tested using **MobileNetV2**. This architecture was chosen due to the following other reasons:

    - MobileNetV2 was recently announced by the researchers at Google. It employs **state of the art** features from all the recent advancements in deep learning architectures. 
  
    - The number of parameters in MobileNetV2 can be varied by the **depth multiplier** alpha. This provides more flexibility in making the tradeoff between speed and performance.
  
    - It has fewer trainable parameters(~4-7 million) when compared to Xception(~20 million) and even fewer when compared to InceptionResnetV2(~55 million)

- The strategy employed in the limited time was to train four MobileNetV2 models. Which will be exposed to different part of the training data by setting different random seeds. Hence, the data will be shuffled randomly before splitting into the train and validation set.

- The four MobileNets achieved **0.88- 0.89 IOU**
  The key differences in these 4 models are:
    - *MobileNet_model1* : seed = 42, image size = 224x224, alpha = 1.4, batch size = 16 and it is a Fully Convolutional NN.
    - *MobileNet_model2* : seed = 84, image size = 160x120, alpha = 1.4, batch size = 32 and it has dense layers.
    - *MobileNet_model3* : seed = 24, image size = 160x120, alpha = 1.4, batch size = 32 and the dense layers have been modified to have 3 GaussianDropout layers.
    - *MobileNet_model4* : seed = 48, image size = 240x180, alpha = 1.6, batch size = 16 and the dense layers were similar to model3.

    Here, *alpha* is the **depth multiplier**.

- We also introduced some functionalities from *keras.callbacks* such as *EarlyStopping* and *ReduceLROnPlateau* in these MobileNet models. EarlyStopping helped in reducing the number of epochs ran based on the stagnation of change in metric. ReduceLROnPlateau helped in better convergence of solution in the later epochs by reducing the learning rate.
- **Stacking** the models in the source file was done by taking a **simple average**. However, while making public LB submissions we were using a weighted average for stacking. Where the weights for each model were the IOU scores obtained for it by the submission solver. However, we can't submit the individual predictions and then take their IOUs as weights for the final source code. 
  
- This issue was faced because of the shortcomings of Approach 1, we had **not implemented the IOU metric** for it. And due to time constraints, this couldn't be done and ran again.
  