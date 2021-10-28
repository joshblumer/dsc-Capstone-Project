# Capstone Project

License Plate Detection Using Mask_RCNN (Mask Region Based Proposal Convolutional Neural Network)

## Introduction

For my Flatiron School Data Science capstone project I used an object detection library known as Mask_RCNN to build a neural network that can detect license plates in photos of vehicles. Automated license plate recognition (ALPR) is a technology that has been implemented by law enforement, building security cameras, parking garage cameras, and several other examples since the 1970's. Traditional ALPR systems use high speed cameras with infrared filters or an added infrared camera to process different wavelengths of light and light density that bounce off of a license plates reflective surface. Once captured this signal is sent to an Optical Character Recognition (OCR) application software to transform the image into alphanumeric characters. The OCR software matches characters by comparing them to a database of stored pixel values, and then the license plate information is sent to another database to be compared to license plate numbers of interest. It is important to note that the components in this technology system are all static, they each perform a specific task requiring a seprate piece of hardware and software, none of which implement artificial intelligence technologies. The ALPR systems used by law enforcement cost between $10,000 to $20,000, and building cameras with much more limited performance cost around $1,000 to $2,000. These systems are also quite cumbersome with many building cameras ranging in size from as small as a softball to as large as a basketball. Due to the cost, size, and complexity of these systems they are not widely used outside of law enforcement and parking lot/garage monitoring, but there is a growing need to implement this kind of technology in a smaller, smarter, and more cost effective format that can be utilized by consumers and businesses. 

A robust machine learning model with the ability to detect and store license plate information has many applications that can help consumers and businesses. The focus of my business use case research was into the implementation of neural network driven license plate detection in vehicle dash cameras. A dashcam is a vehicle mounted camera that records video outside of the vehicle, inside, or both. They are available in a range of video resolutions and features that include monitoring acceleration, deceleration, speed, steering angle, GPS data, and vandalism monitoring as well. Dashcam monitoring has helped vehicle owners mitigate property loss by capturing accident and incident event footage, but there are many qualitative articles and reviews that assert that even with high resolution video, license plate information is not legible. This is a large problem due to up to 90% of hit-and-run events going unsolved. In 2015 there were almost 750,000 hit-and-run crashes that caused over 2,000 fatalities and several millions of dollars in costs of property loss, loss of productivity, and medical bills. Without the ability to identify the at fault driver, most hit-and-run victims are left to pay their own auto and health insurance deductibles and face rate increases, or worse, have to be liable for the full costs if they are un or under-insured. Hit and runs have increased by over 60% in the last decade and continue to increase by 6-7% annually. 

The global dash cam market grew from $1.2 billion in 2012 to $4 billion in 2020, and is projected to continue growing at a rate of 10-12% per year to over $6 billion by 2025. Dashcams are segmented into basic, advanced, and smart categories based on their available features and underlying technologies, with basic and advanced representing over 90% of all dashcam sales. Dashcam popularity continues to rise as consumers realize their potential to help protect their health and property, and as more and more auto insurance companies offer rate discounts to customers who purchase and use them. At this time I was not able to find a dashcam product or service making use of artificial intelligence modeling to capture and store license plate information. Given the increase in numbers of hit-and-runs and auto incidents coupled with the projected growth of the dashcam market and lack of smart cam competition, a company that is able to develop and license an artificial intelligence license detection software or manufacture an artificial intelligence driven device is in a very advantageous position to enter the market and capture a competitive portion of dashcam sales. 

## Technologies
* The data preparation, organization, and modeling for this project can be found in the jupyter notebook file titled 'Mask_RCNN.ipynb', it is located in this page's repo and was created using Python 3.5.6
* There is a keynote presentation for non-technical audiences available under the file name "CapstonePresentation.pdf"

### Necessary libraries to reproduce and run this project are:
* OS
* Pandas
* NumPy
* XML.ETree
* MatPlotLib
* Tensorflow 1.5
* Keras 2.2.4
* MRCNN

The data used for this project was downloaded from Kaggle at this [link](https://www.kaggle.com/andrewmvd/car-plate-detection)

## Objectives

* Explore and analyze dataset images and annotation files
* Preprocess and normalize the files using MRCNN library dataset objects
* Model the image data using a Mask Region based proposal Convolutional Neural Network
* Evaluate the model's performance using the mean Absolute Precision (mAP) score
* Validate model performance by generating predictions on test images with no label

## Methodology



