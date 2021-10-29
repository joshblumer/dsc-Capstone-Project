# Capstone Project

License Plate Detection Using Mask_RCNN (Mask Region Based Proposal Convolutional Neural Network)

## Introduction

For my Flatiron School Data Science Capstone Project I used an object detection library known as Mask_RCNN to build a neural network that can detect license plates in photos of vehicles. Automated license plate recognition (ALPR) is a technology that has been implemented by law enforement, building security cameras, parking garage cameras, and several other examples since the 1970s. Traditional ALPR systems use high speed cameras with infrared filters or an added infrared camera to process different wavelengths of light and light density that bounce off of a license plates reflective surface. Once captured this signal is sent to an Optical Character Recognition (OCR) application software to transform the image into alphanumeric characters. The OCR software matches characters by comparing them to a database of stored pixel values, and then the license plate information is sent to another database to be compared to license plate numbers of interest. It is important to note that the components in this technology system are all static, they each perform a specific task requiring a seprate piece of hardware and software, none of which implement artificial intelligence technologies. The ALPR systems used by law enforcement cost between $10,000 to $20,000, and building cameras with much more limited performance cost around $1,000 to $2,000. These systems are also quite cumbersome with many building cameras ranging in size from as small as a softball to as large as a basketball. Due to the cost, size, and complexity of these systems they are not widely used outside of law enforcement and parking lot/garage monitoring, but there is a growing need to implement this kind of technology in a smaller, smarter, and more cost effective format that can be utilized by consumers and businesses. 

A robust machine learning model with the ability to detect and store license plate information has many applications that can help consumers and businesses. The focus of my business use case research was into the implementation of neural network driven license plate detection in vehicle dash cameras. A dashcam is a vehicle mounted camera that records video outside of the vehicle, inside, or both. They are available in a range of video resolutions and features that include monitoring acceleration, deceleration, speed, steering angle, GPS data, and vandalism monitoring as well. Dashcam monitoring has helped vehicle owners mitigate property loss by capturing accident and incident event footage, but there are many qualitative articles and reviews that assert that even with high resolution video, license plate information is not legible. This is a large problem due to up to 90% of hit-and-run events going unsolved. In 2015 there were almost 750,000 hit-and-run crashes that caused over 2,000 fatalities and several millions of dollars in costs of property loss, loss of productivity, and medical bills. Without the ability to identify the at fault driver, most hit-and-run victims are left to pay their own auto and health insurance deductibles and face rate increases, or worse, have to be liable for the full costs if they are uninsured or under insured. Hit and runs have increased by over 60% in the last decade and continue to increase by 6-7% annually. 

The global dash cam market grew from $1.2 billion in 2012 to $4 billion in 2020, and is projected to continue growing at a rate of 10-12% per year to over $6 billion by 2025. Dashcams are segmented into basic, advanced, and smart categories based on their available features and underlying technologies, with basic and advanced representing over 90% of all dashcam sales. Dashcam popularity continues to rise as consumers realize their potential to help protect their health and property, and as more and more auto insurance companies offer rate discounts to customers who purchase and use them. At this time I was not able to find a dashcam product or service making use of artificial intelligence modeling to capture and store license plate information. Given the increase in numbers of hit-and-runs and auto incidents coupled with the projected growth of the dashcam market and lack of smart cam competition, a company that is able to develop and license an artificial intelligence license detection software or manufacture an artificial intelligence driven device is in a very advantageous position to enter the market and capture a competitive portion of dashcam sales. 

## Technologies
* The data preparation, organization, and modeling for this project can be found in the jupyter notebook file titled 'Mask_RCNN.ipynb', it is located in this page's repo and was created using Python 3.5.6

* There is a keynote presentation for non-technical audiences available under the file name "CapstonePresentation.pdf"

* The data used for this project was downloaded from [Kaggle](https://www.kaggle.com/andrewmvd/car-plate-detection).

* The MS COCO weights that are loaded into the model during training can be found [here](https://github.com/matterport/Mask_RCNN/releases).


### Necessary libraries to reproduce and run this project are:
* OS
* Pandas
* NumPy
* XML.ETree
* MatPlotLib
* Tensorflow 1.5
* Keras 2.2.4
* MRCNN

## Objectives

* Explore and analyze dataset images and annotation files
* Preprocess and normalize the files using MRCNN library dataset objects
* Model the image data using a Mask Region based proposal Convolutional Neural Network
* Evaluate the model's performance using the mean Absolute Precision (mAP) score
* Validate model performance by generating predictions on test images with no label

## Methodology

Computer vision is the field of machine learning concerned with how computers can understand digital images and videos with the goal of automating tasks that are usually completed by a biological visual system. There are many sub domains within computer vision and the most prominent is recognition, which aims to process images and determine if they contain specific objects or features. Object recognition is further divided into image classification, which only assigns a label to an image and image object detection, which draws a box around a specific object and assigns it a label as well. The foundation of both of these tasks is the Convolutional Neural Network, but object detection requires several additional steps. I've given a brief summary on the structures of neural networks and convolutional neural networks in my most recent project, if you're not familiar with them you can reference that topic at this [link](https://github.com/joshblumer/dsc-phase-4-project). 

To advance from image classification to object detection you have to add layers to the data processing and modeling that draw bounding boxes around the kind of objects you want to detect. There are two modeling techniques with different approaches to object detection and they are the R-CNN (Region based CNN) family, and the YOLO (You Only Look Once) family. The YOLO model approach takes an image and splits it into a grid, within each grid it takes a number of bounding boxes and the network outputs a class probability and bounding box values for each bounding box. The bounding boxes that have a class probability that is above a specific threshold are selected and used to locate the object in each image. YOLO models are much faster than RCNNs due to being a single end to end network, they can make predictions starting at 45 frames per second (fps), but they usually generate more localization errors and struggle to detect smaller objects. Given the scope of my project being to detect license plates, which can be small in many images, I chose to implement a version of a RCNN. 

The original RCNN model developed by [Ross Girshick et all](https://arxiv.org/pdf/1311.2524.pdf) utilized a selective search method that extracted 2000 regions from an image called regional proposals. Once generated, the regional proposals are fed into a CNN that extracts their features and outputs a dense layer consisting of the features that are then fed into a Support Vector Machine to classify objects. The original RCNN took a large amount of time to train due to having to classify 2000 region proposals per image which led to the next iterations of the RCNN family, fast, and faster-RCNN. Fast and faster RCNN greatly improved the speed of RCNNs by replacing the initial 2000 region grid with a CNN to generate a convolutional feature map which indentified regions of proposal, used a region of interest pooling layer to compress them, and then a softmax layer to predict the class of the proposed region and perform regression on the bounding box. This reduced image processing time from 49 seconds to 2.3 seconds. The faster RCNN model further improved on the fast model by replacing the selective search algorithm used on the feature map to identify region proposals, with another separate network that again used region of interest pooling, which reduced image processing time from 2.3 seconds to 0.2 seconds. The Mask RCNN model used in my object detection model was built directly on top of a Faster-RCNN framework. 

The Mask RCNN model functions very similarly to a Faster-RCNN model with the exception of adding a third output for each proposed object. In addition to the class label and bounding box offset, the Mask Model also outputs a third branch known as an object mask. The mask requires a finer spatial layout of an object which makes it more efficient at image segmentation. Image segmentation is broken down into semantic segmentation, which categorizes similar objects at the pixel level as a single class (think background vs. foreground) and instance segmentation, which further segments each individual object on similar planes. The addition of a mask that generates a pixel by pixel alignment elevates Mask models to a best in class solution for object detection, and an especially good solution for detecting a small object such as a license plate that is on the same plane as another object (the vehicle it's attached to).

## Table of Contents

* [Exploratory Data Analysis](#EDA)
* [Preprocessing](#Process)
* [Modeling](#Models)
* [Model Evaluation](#Eval)
* [Conclusions](#Conclude)
* [Resources](#Resource)

<a name="EDA"></a> 
## Exploratory Data Analysis

The dataset I used for my object detection model contains 433 images of vehicles with license plates and 433 XML files given in the PASCAL VOC format that contains their bounding box information. I split this dataset into 365 training examples and 68 testing examples. This would be considered a very sparse dataset for a neural network model as they tend to perform better with as many images as possible. When working with a small dataset you can implement image augmentation to provide added variance within the dataset to help reduce the amount the model over-fits. 

<a name="Process"></a>
## Preprocessing 

The first step in preprocessing this dataset was removing the 'Cars' prefix from the image and annotation filenames. The Mask_RCNN library requires that data be read in as a 'utils.Dataset' object which takes your provided directory files and stores them as an indexed object. These indexed objects id's can only be processed as integers so any characters other than numeric have to be removed from the file names. Once the data was prepared to be read in I defined functions to load the image and annotation data, extract the bounding box information from the annotation XML files, generate masks using the bounding box coordinates, and load an image reference path to each image. Image data models usually perform best when the images are normalized and that was performed on these images using the Mask_RCNN libraries 'mold_image()' function by centering the pixel values.

pic

<a name="Models"></a>
## Modeling 






