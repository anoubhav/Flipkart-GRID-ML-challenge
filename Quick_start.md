### Perform the following steps in the given order:
1. The **current working directory** should be set to *Flipkart_ML_Squad_Source*.
   
2. Execute *prep_data_npy_by4_norm_aug.py*
     * Set the directory paths for following constants in the file:
         * IMG_DIR  : contains all images
         * TRAIN_PATH: contains training.csv
         * TEST_PATH : contains test.csv
     * **Note:** By moving training.csv and test.csv directly to *Flipkart_ML_Squad_Source directory*, the default paths in the scripts will work.
     * For IMG_DIR the default path is obtained by creating a directory called *images* in *Flipkart_ML_Squad_Source* and moving the entire image dataset to this *images* directory.
  
     * This file creates 4 numpy arrays.
  
    * **Note**: Restart the kernel after every script to release memory of previous script variables.
    *  **Note**: In the unfortunate event of some of the model files throwing errors on execution. Please ignore those files and continue with the remaining steps. 
  
3.  Execute *Xception_model.py*
    * TEST_PATH: contains test.csv  
    * This file creates a **directory** named *Xception_checkpoints*.
    * It also creates a **csv file** named *predictions_Xception.csv* . 
    * This is just **one of the output files** which will be stacked with other prediction files below. 
 

4.  Execute *InceptionResNetV2_model.py*
    * TEST_PATH: contains test.csv
    * This file creates a **directory** named *InceptionResNetv2_checkpoints*.
    * It also creates a **csv file** named *predictions_InceptionResNetV2.csv* . 
    * **Note**: For this model, it took around 20mins on my system specs to save the model weights(~630 MB) for **only the first epoch**. In the subsequent epochs saving checkpoints was quick.
  
5.  Execute *prep_data_approach2.py*
    * Set the directory paths for following constants in the file:
         * IMG_DIR   : contains all images
         * TRAIN_PATH: contains training.csv
         * TEST_PATH : contains test.csv
     * This file creates **two directories** named *train_images* and *test_images*
     

6.  Execute *MobileNet_model1.py*
    * TEST_PATH: contains test.csv
    * TRAIN_PATH: contains training.csv
    * This file creates a **directory** named *Mobilenet_checkpoints*.
    * It creates **two csv** files named *train_1.csv* and *val_1.csv*
    * It also creates a **csv file** named *predictions_MobileNet_model1.csv* . 
  
7.  Execute *MobileNet_model2.py*
    * TEST_PATH: contains test.csv
    * TRAIN_PATH: contains training.csv
    * It creates **two csv** files named *train_2.csv* and *val_2.csv*
    * It also creates a **csv file** named *predictions_MobileNet_model2.csv* . 
  
8.  Execute *MobileNet_model3.py*
    * TEST_PATH: contains test.csv
    * TRAIN_PATH: contains training.csv
    * It creates **two csv** files named *train_3.csv* and *val_3.csv*
    * It also creates a **csv file** named *predictions_MobileNet_model3.csv* . 

9.  Execute *MobileNet_model4.py*
    * TEST_PATH: contains test.csv
    * TRAIN_PATH: contains training.csv
    * It creates **two csv** files named *train_4.csv* and *val_4.csv*
    * It also creates a **csv file** named *predictions_MobileNet_model4.csv* . 
  
10. Execute *Final_stack_predictions.py*
    * This file outputs **FINAL_PREDICTION.csv**
    * **This is the file I want to submit for the evaluation.**
