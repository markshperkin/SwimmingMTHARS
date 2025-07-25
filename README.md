# Multi-Task Swimming Activity Recognition and Segmentation using Single IMU data

## Overview
This project implements the MTHARS architecture, a multi-task deep learning approach designed to segment and classify IMU data streams. It was developed as part of my M.S. thesis research, where I collected and labeled swimming motion data using a single wrist-worn IMU and my [labeling software](https://github.com/markshperkin/LabelingSoftware).    
The model is trained to detect and classify nine distinct swimming activities, including various stroke phases and underwater movements. The system is optimized for handling real-world, high-frequency sensor data in an end-to-end learning pipeline.     
Swimming data was collected from a single wrist-worn IMU worn by athletes from the [University of South Carolina's Division I SEC swim team](https://gamecocksonline.com/sports/swimming/).       

### [Manuscript](https://github.com/markshperkin/SwimmingMTHARS/blob/main/menuscript.pdf) 

To access the dataset, please email me.  

## Usage

### Data
There are many files to view/analyze/manipulate the dataset to the models' needs.

 - to get the train and val datasets ready for training:
```
python dataCleanerVal.py
python dataCleanerTrain.py
```
 - to view before and clean versions of the data (dataTesterB - before; dataTesterC - Clean):
```
python dataTesterB.py
python dataTesterC.py
```
 - to analyze the dataset and get counts and duration of each label:
```bash
python dataAnalyze.py
```

### Training and evaluating
move to *MTHARS* dir.    
To start training, make sure to comment and uncomment areas in *train.py* that work for accelerometer only data streams and accelerometer & gyroscope data streams.     
```
python Train.py
```
to evaluate the model
```
python evaluate_model.py
```
to plot the model's predictions on a sample
```
python evaluator.py
```
to plot the output training file
```
python plotResults.py
```

for more information, please check out my manuscript at the top of the read me on how everything is working.

## Thesis Project
The thesis was guided by the following University of South Carolina faculty:  
 - [Homayoun Valafar](https://www.sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/homayounvalafar.php)
 - [Ramtin Zand](https://www.sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/zand.php)
 - [Vignesh Narayanan](https://sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/narayanan_vignesh.php)
 - [Forest Agostinelli](https://www.sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/forest_agostinelli.php)


