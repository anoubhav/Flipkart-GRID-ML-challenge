""" Goes through all the downloaded images(56,791) and copies them into separate train images and test images folders.
So that for further analysis all 56,791 images don't need to be screened. The useful images(14000+12815) are separated.
The train images are the further divided so that a few are used for validation and the rest for training. """

# Library imports
import os
import cv2
import pandas as pd


IMG_DIR = os.getcwd() + '\\images'
    
TRAIN_PATH = 'training.csv'
TEST_PATH  = 'test.csv'


# Creates folder for storing train_images
newpath = os.path.join(os.getcwd(), 'train_images')
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
# Creates folder for storing test_images
newpath = os.path.join(os.getcwd(), 'test_images')
if not os.path.exists(newpath):
    os.makedirs(newpath)
    

def create_img_separator():
    training_data = pd.read_csv(TRAIN_PATH)
    testing_data = pd.read_csv(TEST_PATH)

    # For identifying which set image belongs to, i.e. train, test or useless
    train_img_names = list(training_data['image_name'])
    test_img_names  = list(testing_data['image_name'])

    img_counter, error_counter, useless_counter = 1, 0, 0

    for img_name in os.listdir(IMG_DIR):
        print(img_counter,' ---> ', img_name)
        img_counter += 1

        # Image path
        img_path = os.path.join(IMG_DIR, img_name)

        # Error: Assertion failed (ssize.width > 0 && ssize.height > 0) in cv::resize. This happened because file name had non-ASCII characters for some cases. These cases are captured in errors.
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Is image in training set
            if img_name in train_img_names:
                path = os.path.join(os.getcwd()+'\\train_images', img_name)
                cv2.imwrite(path, img)

            # Is image in test set
            elif img_name in test_img_names:
                path = os.path.join(os.getcwd()+'\\test_images', img_name)
                cv2.imwrite(path, img)
                
            # Image not in either train/test
            else:
                useless_counter += 1

        except:
            error_counter += 1
            continue

    # return and save all errors
    print('No. of identified errors:', error_counter)

    # print number of useless images
    print('No. of uselesss images:', useless_counter)
    
    
def main():
    
    input('-'*80+'\n\nPlease ensure that the correct path for images is given to IMG_DIR constant.\n\n'+'-'*80)
    input('-'*100+'\n\nPlease ensure that the correct path for training.csv and test.csv is given to TRAIN_PATH and TEST_PATH.\n\n'+'-'*100)
       
    create_img_separator()
    
if __name__ == "__main__":
    main()







