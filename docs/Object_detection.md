# **Model Selection and Justification**

## Chosen Model: Faster R-CNN with ResNet-50-FPN
For object detection on the BDD100K dataset, I have selected **Faster R-CNN with a ResNet-50 Feature Pyramid Network (FPN)** as the base architecture. The model is initialized with **pre-trained weights**, which has been specifically trained for the BDD100K dataset.

## Why Faster R-CNN?
Faster R-CNN is a **two-stage object detection model** that provides a good balance between accuracy and efficiency. The reasons for choosing this model include:

### **High Detection Accuracy**
- Faster R-CNN is known for its high accuracy in detecting objects, particularly when dealing with complex urban driving scenarios like BDD100K.
- It performs well on **small and occluded objects**, which are common in street scenes (e.g., traffic signs, pedestrians, vehicles).

### **Pretrained Weights on BDD100K**
- Using **pre-trained weights** helps leverage **transfer learning**, reducing training time and improving performance on the dataset.
- The model has already learned relevant **feature representations from BDD100K**, making it more suitable for fine-tuning.

### **Feature Pyramid Network (FPN) for Multi-Scale Detection**
- The **ResNet-50-FPN backbone** enhances detection performance by improving feature extraction across different object sizes.
- This is crucial for detecting **small objects like pedestrians, traffic lights, and signs** in varying lighting conditions.

### **Balance Between Performance and Computational Efficiency**
- Unlike single-stage detectors like YOLO or SSD, Faster R-CNN excels at precise localization and classification, making it a strong candidate for **POCs, Assessments ** scenarios where quality matters more than speed.
- While Faster R-CNN is not the fastest object detector, it provides a good **trade-off between accuracy and inference speed**, which is important for applications like **autonomous driving**.


# **Key Components of Faster R-CNN**

Faster R-CNN builds upon the success of Fast R-CNN by introducing a novel component: the **Region Proposal Network (RPN)**. The RPN allows the model to generate its own **region proposals**, creating an **end-to-end trainable object detection system**. Below are the key components that make Faster R-CNN so effective.

## **1. Backbone Network**
The **backbone network** acts as the **feature extractor** for Faster R-CNN. Typically, this is a **pre-trained Convolutional Neural Network (CNN)** such as **ResNet** or **VGG**. This network processes the input image and generates a **feature map**, which encodes **hierarchical visual information**.

- The feature map has a **smaller spatial size** than the input image but retains **deep channel information**.
- This compact representation is essential for both **region proposal generation** and **object classification** tasks.

## **2. Region Proposal Network (RPN)**
The **RPN** is the core component of Faster R-CNN, responsible for generating **region proposals**. It is a **fully convolutional network** that takes the **feature map** from the backbone network as input.

### **How RPN Works**
- The RPN operates by sliding a small **network over the feature map**.
- At each position in the sliding window, it **predicts multiple region proposals** with associated **classification scores**.
- It introduces the concept of **anchors**—predefined bounding boxes of various **scales** and **aspect ratios** centered at each location in the feature map.

### **For each anchor, the RPN predicts:**
1. **Objectness Score** – The probability that the anchor contains an object of interest.
2. **Bounding Box Refinements** – Adjustments to the anchor’s coordinates to better fit the detected object.

This efficient design allows the RPN to generate high-quality region proposals at multiple **scales and aspect ratios**.

## **3. RoI Pooling Layer**
The **Region of Interest (RoI) pooling layer** is crucial for handling region proposals of **different sizes**. It ensures that all proposals are converted into **fixed-size feature maps**, making them compatible with the classification and regression layers.

### **How RoI Pooling Works**
- Each **region proposal** is divided into a **fixed grid** (e.g., **7×7**).
- A **max-pooling operation** is applied to each grid cell.
- This process ensures that the output **feature map** has a consistent size (e.g., **7×7×512**).

By doing so, RoI pooling allows Faster R-CNN to efficiently process region proposals of different sizes while maintaining computational efficiency.

## **4. Classification and Bounding Box Regression Heads**
The final component of Faster R-CNN consists of **two parallel fully connected layers**:

### **1. Classification Head**
- Uses a **softmax activation** to predict the **object class** for each region proposal.

### **2. Bounding Box Regression Head**
- Further refines the bounding box coordinates for improved localization accuracy.

### **Loss Function**
- **Cross-Entropy Loss** – Optimizes the classification accuracy.
- **Smooth L1 Loss** – Optimizes bounding box regression.

By jointly optimizing both classification and localization, Faster R-CNN effectively detects and localizes objects with high accuracy.

# **The Architecture of Faster R-CNN**  

Faster R-CNN unifies multiple components into a **single network** for efficient object detection. The **workflow** is as follows:  

1. The **input image** is processed through the **backbone CNN**, generating a **feature map**.  
2. This feature map is passed to the **Region Proposal Network (RPN)** and **RoI Pooling Layer**.  
3. The **RPN** scans the image with **anchor boxes**, proposing regions based on **classification scores**.  
4. The **RoI pooling layer** refines these region proposals and prepares them for classification.  
5. The **Classification Head** predicts the **object class** for each proposal.  
6. The classification data is then fed into the **Bounding Box Regression Head**, which further **refines coordinates** to yield the **final detection output**.  

This unified design makes Faster R-CNN a highly efficient and accurate object detection model.


# Model Performance Evaluation

## Metrics : Precision, Recall, F1-Score, Mean Average Precision (mAP)
### Reason of choosing:

1. Precision ensures that detected objects are actually correct (minimizes false positives).

2. Recall measures how many actual objects were detected (minimizes false negatives).

3. F1-Score balances both Precision and Recall, providing a single performance measure that accounts for both correctness and completeness.

4. mAP (Mean Average Precision) is chosen because it balances Precision and Recall across different IoU thresholds, ensuring both correct detections and accurate localization. It also provides an overall performance measure by averaging precision across all object classes. 

## Quantitative Evaluation
### 1. Precision, Recall & F1-Score Across IoU Thresholds

| IoU Threshold | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| 0.1           | 0.8197    | 0.9908 | 0.8972   |
| 0.2           | 0.7854    | 0.9879 | 0.8751   |
| 0.3           | 0.7641    | 0.9394 | 0.8427   |
| 0.4           | 0.7209    | 0.8678 | 0.7876   |
| 0.5           | 0.6700    | 0.7402 | 0.7034   |
| 0.6           | 0.5685    | 0.6614 | 0.6114   |
| 0.7           | 0.4320    | 0.5245 | 0.4738   |
| 0.8           | 0.2701    | 0.3275 | 0.2960   |
| 0.9           | 0.0990    | 0.1219 | 0.1093   | 

>  **Observations**: The model demonstrates high recall across lower IoU thresholds, with recall peaking at 0.9908 (IoU 0.1), indicating that it detects most objects but with some localization imprecision. As IoU increases, precision drops significantly (0.8197 → 0.0990), showing that stricter overlap criteria reduce confident detections. The F1-score remains strong up to IoU 0.5 (0.7034) but declines sharply beyond, suggesting room for improvement in bounding box refinement and localization accuracy.


### 2. mAP & Class wise AP 


| IoU  | mAP    | 1      | 2     | 3       | 4     | 5     | 6     | 7      | 8      | 9     |  10    |
|------|--------|--------|-------|--------|-------|--------|------|--------|--------|-------|--------|
| 0.5  | 0.408  | 0      | 0.571 | 0.679   | 0.561 | 0.377 | 0.597 | 0.3639 | 0.6098 | 0.0000 | 0.3204 |

>  **Observations**: The model achieves a mAP of 0.408 at IoU 0.5, with strong detection of common objects like traffic signs (0.679) and cars (0.6098). However, it struggles with underrepresented classes such as buses (0.0) and trains (0.0), leading to poor generalization. Enhancing dataset balance and optimizing anchor sizes could improve detection for these weaker categories.  

#### Class to Index Mapping

Mapping of object classes to their corresponding index values. This mapping is used in the entire code and documentation.

Class | Bus | Traffic Light | Traffic Sign | Person | Bike | Truck | Motor | Car | Train | Rider |
------|-----|--------------|--------------|--------|------|-------|-------|-----|-------|-------|
Index |  0  |      1       |      2       |   3    |  4   |   5   |   6   |  7  |   8   |   9   |


## Qualitative Evaluation

### What Works for the Model
- The model performs well in detecting common objects like **cars, traffic lights, signboards, and pedestrians**.
- **Cars** are detected with high confidence due to their large presence in the training data.
- **Traffic lights and signboards** are also consistently recognized, even in varying lighting conditions.
- The model effectively handles **pedestrians**, even when they are partially occluded.

### What Does Not Work for the Model
#### Bus and Truck Detection Issues:
- The model struggles to detect **buses and trucks**, likely due to their **lower representation** in the dataset compared to cars.
- The model may **misclassify buses as large cars** or fail to detect them altogether.

#### Poor Detection of Trains, Riders, and Motorcycles:
- **Train detection is almost absent**, likely due to the **scarcity of train instances** in the dataset.
- **Riders (motorcycle/bicycle riders) and motorcycles** have **limited prediction visualizations**, indicating weaker detection capabilities.
- This may be due to their **relatively small size in images**, making it harder for the model to learn effective features.

### Possible Improvements
- **Data Augmentation**: Increasing the representation of underrepresented classes using data augmentation techniques like **oversampling, synthetic data generation, or augmenting existing images**.
- **Class-Specific Fine-Tuning**: Training the model with a **balanced dataset**, ensuring that less frequent objects get enough training samples.
- **Anchor Box Optimization**: Adjusting **anchor sizes** to **better detect smaller objects** like riders and motorcycles.
- **underrepresented categories** learning can improve using class balancing techniques.


## Conclusion
The Faster R-CNN model performs well in detecting frequently occurring objects like cars, traffic lights, and pedestrians but struggles with underrepresented classes such as buses, trucks, trains, riders, and motorcycles. These challenges stem from dataset imbalance and difficulties in detecting small or rare objects. By leveraging data augmentation, class-specific fine-tuning, and anchor box optimization, detection performance can be improved. This evaluation provides a solid foundation for refining the model and preparing it for real-world deployment.