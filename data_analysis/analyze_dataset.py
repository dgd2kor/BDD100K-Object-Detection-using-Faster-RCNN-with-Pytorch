import os
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

from parse_json import load_annotations, extract_objects

def category_count(json_file):
    """
    Parses a JSON annotation file, extracts object categories, and computes their occurrence counts.

    Args:
        json_file (str): Path to the JSON file containing dataset annotations.

    Returns:
        pd.DataFrame: A DataFrame with two columns - 'Category' and 'Count', 
                      listing the unique object categories and their respective counts in descending order.
    """
    # Load, process, and save
    annotations = load_annotations(json_file)
    categories, _ = extract_objects(annotations)

    category_counts = Counter(categories)
    df_count = pd.DataFrame(category_counts.items(),columns=['Category','Count'])
    df_count.sort_values(by='Count',ascending=False,inplace=True)
    df_count = df_count.reset_index(drop=True)

    return df_count

def distribution_analysis(train_df, val_df):
    """
    Analyzes the distribution of object categories in the training and validation datasets.

    Args:
        train_df (pd.DataFrame): DataFrame containing category counts for the training dataset.
                                 Expected columns: ["Category", "Count"].
        val_df (pd.DataFrame): DataFrame containing category counts for the validation dataset.
                               Expected columns: ["Category", "Count"].

    Returns:
        pd.DataFrame: A DataFrame with merged category distributions across train and validation sets.
                      Includes total count and percentage distribution per split.
                      Columns: ["Category", "train_count", "val_count", "Total_count",
                                "train_distribution in %", "val_distribution in %"].
    """
    train_df.columns = ["Category", "train_count"]
    val_df.columns = ["Category", "val_count"]

    df_merged = pd.merge(train_df, val_df, on="Category", how="outer").fillna(0)
    df_merged["Total_count"] = df_merged["train_count"]+df_merged["val_count"]
    df_merged["train_ditribution in %"] = ((df_merged["train_count"]/df_merged["Total_count"])*100).round(2)
    df_merged["val_ditribution in %"] = ((df_merged["val_count"]/df_merged["Total_count"])*100).round(2)

    return df_merged


def get_unique_images(data, rare_objects=None, crowded_scene_objects_threshold=50, large_bb_threshold=200000):
    """
    Identifies unique images based on specific characteristics such as rare objects, 
    crowded scenes, and large bounding boxes.

    Args:
        data (list): A list of image annotations, where each entry contains image metadata 
                     and object labels.
        rare_objects (list, optional): A list of object categories considered rare. 
                                       Defaults to ["train", "motor"].
        crowded_scene_objects_threshold (int, optional): The threshold for considering an 
                                                         image as a crowded scene based 
                                                         on the number of objects. 
                                                         Defaults to 50.
        large_bb_threshold (int, optional): The threshold for detecting large bounding boxes 
                                            (outliers) based on pixel area. Defaults to 200,000.

    Returns:
        dict: A dictionary with lists of image names and their corresponding labels categorized as:
              - "rare_objects": Images containing rare object categories.
              - "crowded_scenes": Images with object count exceeding the threshold.
              - "large_bbox": Images containing unusually large bounding boxes.
              - "_labels" keys store the corresponding annotations for each category.
    """
    if rare_objects is None:
        rare_objects = ["train", "motor"]
    unique_images = {
        "rare_objects": [],
        "crowded_scenes": [],
        "large_bbox": [],
        "rare_objects_labels": [],
        "crowded_scenes_labels": [],
        "large_bbox_labels": []
    }

    for entry in data:
        image_name = entry["name"]
        labels = entry["labels"]

        bbox_sizes = []
        num_objects = len(labels)

        for label in labels:
            category = label["category"]
            if category in rare_objects:  # Rare objects
                unique_images["rare_objects"].append(image_name)
                unique_images["rare_objects_labels"].append(labels)

            if "box2d" in label:  # Bounding box exists
                bbox = label["box2d"]
                width = bbox["x2"] - bbox["x1"]
                height = bbox["y2"] - bbox["y1"]
                bbox_sizes.append(width * height)

        # Crowded scenes: More than thresholded objects
        if num_objects > crowded_scene_objects_threshold:
            unique_images["crowded_scenes"].append(image_name)
            unique_images["crowded_scenes_labels"].append(labels)

        # Large bounding boxes (outliers)
        if bbox_sizes and max(bbox_sizes) > large_bb_threshold:  # Adjust threshold as needed
            unique_images["large_bbox"].append(image_name)
            unique_images["large_bbox_labels"].append(labels)
    
    return unique_images

def get_occluded_truncated(annotations):
    """
    Identifies and extracts images containing truncated or occluded objects.

    Args:
        annotations (list): List of image annotations.

    Returns:
        dict: Dictionary with lists of images containing occluded or truncated objects,
              along with their corresponding filtered labels.
    """
    occluded_truncated_sample = {
        "truncated_samples": [],
        "truncated_samples_labels": [],
        "occluded_samples": [],
        "occluded_samples_labels": [],
    }

    for entry in annotations:
        image_name = entry["name"]
        labels = entry["labels"]

        truncated_labels = []
        occluded_labels = []

        for label in labels:
            if label.get("attributes", {}).get("truncated", 0) == 1:
                truncated_labels.append(label)  # Add only truncated objects
            if label.get("attributes", {}).get("occluded", 0) == 1:
                occluded_labels.append(label)  # Add only occluded objects

        if truncated_labels:
            occluded_truncated_sample["truncated_samples"].append(image_name)
            occluded_truncated_sample["truncated_samples_labels"].append(truncated_labels)

        if occluded_labels:
            occluded_truncated_sample["occluded_samples"].append(image_name)
            occluded_truncated_sample["occluded_samples_labels"].append(occluded_labels)

    return occluded_truncated_sample
