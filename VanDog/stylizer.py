import os
import gc
import tensorflow as tf
import sys
import numpy as np
import random
import skimage.io
import logging
import cv2

from Cartoonizer.test_code import cartoonize

file = "Mask_RCNN/mrcnn/model.py"
path = os.getcwd()+file

ROOT_DIR =  os.getcwd()
print('W-- from stylizer - current working directory', ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from Mask_RCNN.mrcnn.config import Config
# import Mask_RCNN.mrcnn.model as modellib


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  # To find local version
import coco

# Directory of images to run detection on
# MASK_PATH = os.path.join("static", "mask.png")
# CARTOON_PATH = os.path.join("static", "cartoon.png")
# MASK_WITH_BG_PATH = os.path.join("static", "mask_background.png")

# OUTPUT_PATH = os.path.join("static", "output.png")
# Test random names for output image
# because after deploying online, we noticed
# that 2 users using the app at the same time
# will overwrite each other's outputs
random_name = 'VanDog_' + ''.join([random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for i in range(10)]) + '.png'
OUTPUT_PATH = os.path.join("static", random_name)

##Testing some static images
# PET_IMG_PATH = os.path.join("static", "Cats.jpg")
# BG_IMG_PATH = os.path.join("static", "Grass.jpeg")

model_dir = os.path.join(ROOT_DIR, "logs")

class InferenceConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 81


class VanDog:
    def __init__(self, segmenter, cartoonizer_path):
        logging.info("Stylizer class initialized")
        # self.segmenter = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)
        # self.segmenter.load_weights(segmenter_path, by_name=True)
        # self.segmenter.keras_model._make_predict_function()
        # Local path to trained weights file
        self.segmenter = segmenter
        self.cartoonizer_path = cartoonizer_path
        logging.info("Models are loaded!")
        
    def get_masks(self, image):
        # IMAGE_PATH = os.path.join(OUR_IMAGES_DIR, image_name)
        # image_path = skimage.io.imread(image_path)
        # image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        logging.info('WIS-CHECKPOINT 1A - Running segmentation detection')

        # Run detection
        print('Segmenter in use:', self.segmenter)
        r = self.segmenter.detect([image], verbose=0)[0]
        # self.segmenter = None
        logging.info('WIS-CHECKPOINT 1B - Segmentation detection done, segmenter deleted')


        mask = r['masks']
        mask = mask.astype(int)

        logging.info('WIS-CHECKPOINT 1C')

        
        detected = []

        num_pets = len(['x' for i in range(mask.shape[2]) if class_names[r['class_ids'][i]] in classes_of_interest])
        print(f"Detected {num_pets} pet(s) / {mask.shape[2]} classes")
        print('Classes detected:', [class_names[r['class_ids'][i]] for i in range(mask.shape[2])])

        print('Results:', r['class_ids'])

        logging.info('WIS-CHECKPOINT 1D')

        for i in range(mask.shape[2]):
            if class_names[r['class_ids'][i]] in classes_of_interest:
                print(class_names[r['class_ids'][i]])
            
                temp = image
                temp = np.where(temp == 0, 1, temp)

                for j in range(temp.shape[2]):
                    temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
                detected.append(temp)
                    
        logging.info('WIS-CHECKPOINT 1E')
        
        # Garbage collection
        # del r
        # del mask

        return detected

    def get_stacked(self, images):
        logging.info('WIS-CHECKPOINT 2-3')

        stacked = np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype='int')
        for i in images:
            stacked += np.where(stacked > 0, 0, 1) * i

        return stacked

    def get_segmented(self, image):
        return self.get_stacked(self.get_masks(image))

    def get_cartoonized(self, image):
        logging.info('WIS-CHECKPOINT 4')
        return cartoonize.cartoonize(image, self.cartoonizer_path)

    def get_stylized(self, pet_filepath, bg_filepath, output_filepath):
        image = skimage.io.imread(pet_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        max_small_dim = 512
        h, w = image.shape[0], image.shape[1]
        if min(h, w) > max_small_dim:
            if h > w:
                h, w = int(max_small_dim*h/w), max_small_dim
            else:
                h, w = 720, int(max_small_dim*w/h)


        image = cv2.resize(image, (w, h))

        bg_image = skimage.io.imread(bg_filepath)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))

        ## The whole thing in one
        # No longer storing in variable to preserve space
        # segmented = self.get_segmented(image)
        # mask_with_bg = self.get_stacked([segmented, bg_image])
        # segmented_and_cartoonized = self.get_cartoonized(mask_with_bg)
        # cv2.imwrite(output_filepath, segmented_and_cartoonized)

        cv2.imwrite(output_filepath, self.get_cartoonized(self.get_stacked([self.get_segmented(image), bg_image])))
        logging.info('Output saved to:', output_filepath)

        # Garbage collection
        # collected = gc.collect()
        # logging.warn("From stylizer.py - Garbage collector: collected",
        #     "%d objects." % collected)
        # return segmented_and_cartoonized

def main():
    logging.info(f"Wis--From stylizer main")
    pass



# Load weights trained on MS-COCO

## From original demo file
# COCO Class names
# Index of the class in the list is its ID.
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# The model sometimes misclassifies cats and dogs as other animals
# We bypass this by adding them to our classes of interest
# In general, this should not affect the result of the segmentation itself
classes_of_interest = ['cat', 'dog', 'sheep', 'teddy bear']

## Wissam note - Changed location of images folder 
## to find our pet images easier 
OUR_IMAGES_DIR = os.path.join(ROOT_DIR, "our_images")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
