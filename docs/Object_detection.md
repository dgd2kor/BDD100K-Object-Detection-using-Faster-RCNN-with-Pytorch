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