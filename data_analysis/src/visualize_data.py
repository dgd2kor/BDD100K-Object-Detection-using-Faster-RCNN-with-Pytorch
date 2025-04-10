import os
import cv2

from matplotlib import pyplot as plt

from analyze_dataset import get_unique_images, get_occluded_truncated
from parse_json import load_annotations
from data_config import Config

config = Config()

def visualize_image(image_dir,image_name,labels,tag=""):
    """
    Visualizes an image with bounding boxes and category labels.
    
    Args:
        image_dir (str): Directory containing the image.
        image_name (str): Name of the image file.
        labels (list): List of object annotations for the image.
        tag (str, optional): Category tag for the image. Defaults to "".    
    Returns:
        None: Displays the image with overlaid bounding boxes.
    """
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for label in labels:
#         print(label)
        if 'category' in label and 'box2d' in label:
            bbox = label['box2d']
            class_ = label['category']
            x1, y1, x2, y2 = map(int, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
            
            cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 2)
            cv2.putText(img, class_, (x1, max(10, y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    image_name_only = str(image_name).split(".")[0]
    folder_name = os.path.join(config.visuals_output_path,tag)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_path = folder_name + f"/{image_name_only}_vis.jpg"
    cv2.imwrite(output_path, img)

def visualize_unique_images(train_image_path, train_annotations, num_visuals):
    """
    Visualizes unique images categorized as rare objects, crowded scenes, and large bounding boxes.
    
    Args:
        train_image_path (str): Directory path to training images.
        train_annotations (dict): Annotations for training images.
        num_visuals (int): Number of images to visualize per category.
    
    Returns:
        None: Displays the visualizations.
    """
    rare_objects = ["train", "motor"]
    crowded_scene_objects_threshold = 50
    
    # BDD100K images have a resolution of 1280 × 720 pixels.
    # The total area per image = 921,600 pixels (1280 × 720).
    # 20% of the image = 921,600 × 0.2 ≈ 184,320 pixels. let's consider 200,000 pixels as threshold
    large_bb_threshold = 200000

    unique_images_in_train = get_unique_images(train_annotations,
                                              rare_objects=rare_objects,
                                              crowded_scene_objects_threshold=crowded_scene_objects_threshold,
                                              large_bb_threshold=large_bb_threshold)

    for r in range(num_visuals):
        r_images, r_labels = unique_images_in_train["rare_objects"][r],unique_images_in_train["rare_objects_labels"][r]
        visualize_image(train_image_path, r_images, r_labels,tag="rare_images")

        c_images, c_labels = unique_images_in_train["crowded_scenes"][r],unique_images_in_train["crowded_scenes_labels"][r]
        visualize_image(train_image_path, c_images, c_labels, tag="crowded_scenes")

        l_images, l_labels = unique_images_in_train["large_bbox"][r],unique_images_in_train["large_bbox_labels"][r]
        visualize_image(train_image_path, l_images, l_labels, tag="large_bbox")


def visualize_occluded_truncated(train_image_path, train_annotations, num_visuals):
    """
    Visualizes occluded and truncated objects in the dataset.
    
    Args:
        train_image_path (str): Directory path to training images.
        train_annotations (dict): Annotations for training images.
        num_visuals (int): Number of occluded/truncated images to visualize.
    
    Returns:
        None: Displays the visualizations.
    """
    occluded_truncated_samples = get_occluded_truncated(annotations=train_annotations)
    for i in range(num_visuals):
        t_images, t_labels = occluded_truncated_samples["truncated_samples"][i],occluded_truncated_samples["truncated_samples_labels"][i]
        visualize_image(train_image_path, t_images, t_labels, tag="truncated")

        o_images, o_labels = occluded_truncated_samples["occluded_samples"][i],occluded_truncated_samples["occluded_samples_labels"][i]
        visualize_image(train_image_path, o_images, o_labels, tag="occluded")

def visualize_all_groundtruths(train_image_path, train_annotations, num_visuals):
    """
    """
    images = []
    train_labels = []
    for entry in train_annotations:
        image_name = entry["name"]
        labels = entry["labels"]

        images.append(image_name)
        train_labels.append(labels)
    for i in range(num_visuals):
        visualize_image(train_image_path, images[i], train_labels[i], tag="all_training_visuals")
    
