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

![bbox](https://raw.githubusercontent.com/joshblumer/dsc-Capstone-Project/master/Photos/bbox.png)
        
* A dataset image displaying it's mask coordinates generated using the information extracted from it's corresponding annotation file
        
![mask](https://raw.githubusercontent.com/joshblumer/dsc-Capstone-Project/master/Photos/label.png)
        
* An example of an image displaying it's mask, bounding box, and label

<a name="Models"></a>
## Modeling 

The Mask_RCNN library is built on top of Tensorflow and Keras and there are similarities in it's implementation of building a CNN, but there are differences as well. The Mask_RCNN model is built using a model configuration object that lets the user specify the number of classes (objects you want to detect) and the number of steps per epoch. Once the configuration class object is defined you instantiate the newly configured object and define the model using Mask_RCNN library specific arguments. From that point the remainder of the training process follows standard CNN steps. Rather than configuring a model architecture from scratch I followed the Mask_RCNN library author's advice and trained the model using downloaded weights from the MS-COCO project (Microsoft Common Objects in Context). The MS-COCO dataset contains 80 classes so it's important to note that you need to exclude class specific output layers when loading the model weights. I trained the model with the default learning rate of 0.001 and 5 epochs and due to the large batch size of 365 (number of training images), the model execution was very time consuming and required 16 hours to train. The Mask_RCNN mmodel outputs several metrics after each epoch, but the only metrics we're concerned with are model loss, class loss, and bounding box loss. The first epoch reported 1.329 train loss/ 1.249 val loss, 0.033 train mrcnn_class_loss/ 0.039 val mrcnn_class_loss, and 0.285 train mrcnn_bbox_loss/ 0.334 val mrcnn_bbox_loss. By the fifth epoch each of those loss metrics decreased to 0.557 train loss/ 0.971 val loss, 0.027 train mrcnn_class_loss/ 0.023 val mrcnn_class_loss, and 0.114 train mrcnn_bbox_loss/ 0.195 val mrcnn_bbox_loss. I chose to freeze all of the MS-COCO layers except for the heads due to time and processing power constraints, but with more time I would have liked to implement a two-stage process that trained the model on frozen weights for one epoch and then fine tuned the model on all layers with none frozen. The decrease in loss with each epoch indicates that that the model is stable though and would have continued to optimize over more epochs. The following visual shows a summary of the model configuration used. 

![model](https://raw.githubusercontent.com/joshblumer/dsc-Capstone-Project/master/Photos/modelconfig.png)
        
* The model configuration used for training, downloaded from the the Mask_RCNN GitHub library

<a name="Eval"></a>
## Model Evaluation

Object recognition and segmentation tasks use their own evaluation metric known as mean absolute precision, (mAP). The task of object recognition is to predict a bounding box around the object you want to detect, so to evaluate how well a model can perform that task you use a metric that calculates how much the predicted bounding box and ground truth bounding box overlap. This is known as intersection over union (IoU) and you calculate it by dividing the area of how much the boxes overlap by the total area of the box. Many classification evaluation metrics use 0.5 as the threshold of a random guess probablitity and 1.0 as the upper limit of classification confidence, and mAP uses those same thresholds as well. The precision in mAP refers to the percentage of bounding boxes that are predicted correctly out of all of the predicted bounding boxes and the mean absolute precision is the mean of all of the precision scores for each value of recall, which is the percentage of correctly predicted bounding boxes out of each object in each image. My transfer learning model using the MSCOCO weights returned a training mAP of 86.50% and testing mAP of 87.50%. I would have liked to return a score over 90%, but given the time and resource constraints, 87.50% is a promising score, and still much higher than 50% which indicates strong and reliable model performance. It is unusual for the test score to be higher than the train score and often indicates that the model wasn't fully trained, but could also be due to variance in the images as the train and test sets were both small.

![train](https://raw.githubusercontent.com/joshblumer/dsc-Capstone-Project/master/Photos/train.png)
        
* Three examples of training images with corresponding mask information, and their bounding box predictions
        
1[test](https://raw.githubusercontent.com/joshblumer/dsc-Capstone-Project/master/Photos/test.png)
        
* Three examples of testing images that had no corresponding mask information, and their bounding box predictions

<a name="Conclude"></a>
## Conclusions


The Mask-RCNN performed very well given a small dataset and the time & resource restrictions that made fully training the model infeasible. Training the model over 5 epochs took around 16 hours on my laptop which is a substantial amount of time to render your hardware unusable while working on any project with a deadline. The loss was improving with each epoch which indicates that it would have continued to improve given more epochs to train through. It wouldn't be feasible for me to increase the number of epochs or attempt the two-stage training on my personal computer, but I would like to use a powerful remote server to execute both of those examples on in the future. 

This model is not yet ready to be put into production, it needs another model added on top of the one I've trained to extract the alphanumeric characters from the license once the license is isolated. But it's strong results clearly showcase a proof of concept that can be incorporated into many types of cameras ranging from dashcams to security cameras and business monitoring cameras. A dashcam complemented by an object recognition and detection algorithm could capture and store license plate information in the event of an accident or incident, potentially saving auto owners hundreds to thousands of dollars or more in the event of a hit and run. A home security camera able to detect and store license plate information could provide home owners and law enforcement with an exact vehicle present during a crime rather that just a make and model, giving precise information about it's owner that could be cross-referenced with other information law enforcement may have about where the vehicle may have been recently. A camera positioned to survey the parking lot of a business with the ability to capture and store license plate information would give business owners precise demographic information about it's customers as well as their shopping habits, enabling more targeted and efficient marketing and advertising campaigns and the ability to build databases with that information for further analysis. It's not clear what the limit is to the potential applications of implementing license plate detection would be, but it is very clear that it is a technology with potential to capture sales, propel market growth, and help businesses and consumers protect themselves and their property.

<a name="Resource"></a>
## Resources 

Hit and Run Statistics:

* https://aaafoundation.org/hit-and-run-crashes-prevalence-contributing-factors-and-countermeasures/

* https://www.farrin.com/blog/shocking-facts-about-hit-and-run-crashes/

* https://www.psychreg.org/percentage-hit-and-run-solved/

Dashcam Market Statistics:

* https://www.grandviewresearch.com/industry-analysis/dashboard-camera-market#:~:text=The%20global%20dashboard%20camera%20market%20size%20was%20estimated%20at%20USD,USD%203.61%20billion%20in%202021.&text=The%20global%20dashboard%20camera%20market%20is%20expected%20to%20grow%20at,USD%208.47%20billion%20by%202028 

* https://www.statista.com/statistics/675288/dashboard-camera-market-size-worldwide/ 
        
ALPR & OCR:

* https://kintronics.com/how-alpr-works/ 

* https://en.wikipedia.org/wiki/Optical_character_recognition 

* https://www.eff.org/pages/automated-license-plate-readers-alpr 

Mask_RCNN References:

* https://github.com/matterport/Mask_RCNN 

* https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/ 

* https://viso.ai/deep-learning/mask-r-cnn/
