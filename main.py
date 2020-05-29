from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from imutils import paths
from utils.resize import image_resize
import numpy as np
import cv2
import shutil
import os
import argparse
import imutils
import time


def range_of_interest(image_of_jersey,
                      lower=[180, 180, 180],
                      upper=[255, 255, 255]):
    """Required parameters: image_of_jersey, lower, upper

       This function takes in an image of a basketball player
       wearing their jersey. It also has predefined threshold variables
       lower and upper that are used to find the largest contours of a given
       RBG color. lower is the lowest RGB value and upper is the highest. They
       are predefined to represnt the threshold of finding the largest WHITE
       contour.

       This function uses OpenCV to isolate and return the basketball players 
       jersey number.
    """

    # convert lists to np array
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find white pixels in the image
    mask = cv2.inRange(image_of_jersey, lower, upper)
    output = cv2.bitwise_and(image_of_jersey,
                             image_of_jersey,
                             mask=mask)

    # apply binary threshold and find contours
    thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # draw box around ROI and extract it (if contours is not 0)
    if len(contours) != 0:
        cv2.drawContours(output, contours, -1, 255, 3)
        x, y, w, h = cv2.boundingRect(c)
        roi = image_of_jersey[y: y + h, x:x + w]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return roi


def classify(jersey_number_roi):
    """Required parameters: jersey_number_roi

       classify accepts the parameter jersey_number_roi.
       It receives this parameters from the function range_of_interest.

       Using a pretrained Keras model, classify returns a string named 'label'
       that is a prediction of the class for the given jersey number.
    """

    # load in trained LeNet model
    test_model = load_model(model_filepath)

    """resize array to 28 width by 28 height and 3 color channels 
       using resize method from utils directory"""
    roi = cv2.resize(jersey_number_roi, (28, 28))
    x = img_to_array(roi)
    x = np.expand_dims(x, axis=0)

    # run prediction on image
    preds = test_model.predict_classes(x)
    prob = test_model.predict_proba(x)
    label = str(preds[0])

    return label


def directory_sort(label, path):
    """Required parameters: label, path.

       directory_sort accepts the parameters label and path.
       label is returned from the function classify.
       path is the original path of the image being classified.

       Using label and path, directory_sort creates a new directory path
       for the images being classified. It uses the label provided by the
       Keras model to sort the images based on the prominate jersey number 
       that was identified in the image.
    """

    # map binary labels to class labels (1, 2, 3, 4, 8)
    Labels = {
        "0": "1",
        "1": "2",
        "2": "3",
        "3": "4",
        "4": "8"
    }

    # if the label is found in Labels, use mapped value
    if label in Labels.keys():
        label = Labels[label]

    # split original image path to create new path
    head_tail = os.path.split(path)
    sorted_dir = "sorted"
    tail = head_tail[1]
    head = head_tail[0]
    new_path = head + "/" + sorted_dir + "/" + label + "/" + tail

    # handle if program is ran on sorted directory
    if "sorted" in path:
        print("[INFO] Images sorted")
        exit()

    # move image to new_path
    shutil.move(path, new_path)


def main(directory_paths):
    """Required parameters: directory_paths.

    This function takes in the input directory of images,
    iterates through the images, extracts the
    jersey number from the given image using
    the range_of_interest function, classifies the jersey number
    using the classify function, and then finally sorts the images
    using the directory_sort function.
    """

    # iterate through input directory
    for path in directory_paths:
        # read image
        image = cv2.imread(path)

        # extract ROI of jersey
        jersey_roi = range_of_interest(image)

        # extract ROI of jersey number
        jersey_number_roi = range_of_interest(jersey_roi,
                                              upper=[190, 140, 190],
                                              lower=[20, 0, 20])

        # identify label for jersey number
        label = classify(jersey_number_roi)

        # sort the original image in new directory
        directory_sort(label, path)

if __name__ == '__main__':
    # set variables for pretrained model and directory of unsorted images
    model_filepath = './model/9923_val-acc.model'
    input_dir = "./demo_dataset/"

    # convert input_dir to a list of paths using the paths method
    directory_paths = list(paths.list_images(input_dir))

    # begin classifying input images and sorting them
    print("[INFO] Classifying images...")
    start = time.time()
    main(directory_paths)
    end = time.time()
    print("[INFO] Time to extract ROIs and "
          "classify them = {} seconds".format(end - start))
