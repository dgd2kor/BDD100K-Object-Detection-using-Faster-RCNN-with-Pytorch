# BDD100K Object Detection using Faster R-CNN with PyTorch

## Overview
This repository implements an object detection pipeline on the **BDD100K** dataset using **Faster R-CNN**. The model leverages **pretrained Faster R-CNN ResNet-50 FPN weights** to detect multiple object categories of **BDD100K Dataset**. This repository includes:
- BDD100K Custom Dataset Loader & Annotation Parser
- End-to-End Training Pipeline with Efficient Data Handling
- Confidence-Thresholded Inference for Precise Predictions
- Post-Inference Bounding Box Visualization for Model Interpretation
- Comprehensive Evaluation using mAP and IoU Analysis
- Structured JSON-Based Prediction Storage for Seamless Processing



## Setup

### **1. Clone Repository**
```sh
git clone https://github.com/dgd2kor/BDD100K-Object-Detection-using-Faster-RCNN-with-Pytorch.git
cd BDD100K-Object-Detection-using-Faster-RCNN-with-Pytorch
```
### **2. Install Dependencies**
```sh
pip install -r requirements.txt
```
### **3. Prepare the Dataset**
Download the **BDD100K dataset (images & labels)** from the official source:

- **Images**: [BDD100K Official Website](https://bdd-data.berkeley.edu/)
- **Labels**: [Download Labels](https://bdd-data.berkeley.edu/)

Once downloaded, extract the dataset and place in data folder and update the data related paths in data_config.py in data_analysis and config.py in object_detection.  

Download and Update the Base weights path in config.py in object detection


## Usage

### **1. Train the Model**
It trains the model with the given hyper parameters in config file.
```bash
python object_detection/train.py
```

### **2. Run Inference**
Generate predictions and save in the output folder.
```bash
python object_detection/inference.py
```

### **3. Visualize Predictions**
Apply prediction boxes and groundtruth boxes on images and stores the visuals in output folder.
```bash
python object_detection/visualizer.py
```
### **4. Metric Evaluator**
Calculates Metrics by loading the saved predictions.
```bash
python object_detection/metrics_evaluator.py
```


