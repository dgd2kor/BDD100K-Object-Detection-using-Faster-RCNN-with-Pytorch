import os
import json
import pandas as pd


def load_annotations(json_path):
    """
    Loads the BDD100K JSON annotations.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: A list of image annotations.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_objects(annotations):
    """
    Extracts object details (class, bounding box) from the JSON annotations.

    Args:
        annotations (list): List of annotations from BDD100K dataset.

    Returns:
        categories (list): list of objects with various categories.
        df_records (pd.DataFrame): A DataFrame containing image file names and object details.
    """
    categories = []
    records = []
    for entry in annotations:
        image_name = entry["name"]
        for label in entry['labels']:
            if "category" in label:
                categories.append(label['category'])
            if "box2d" in label:  # Object detection labels only
                obj_class = label["category"]
                x1, y1, x2, y2 = label["box2d"].values()
                records.append([image_name, obj_class, x1, y1, x2, y2])
    columns = ["Image name","Class", "Bbox x1","Bbox y1", "Bbox x2", "Bbox y2"]
    df_records = pd.DataFrame(records,columns=columns)

    return categories, df_records

def save_to_csv(df, output_path):
    """
    Saves the extracted object data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame with parsed object details.
        output_path (str): Path to save the CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
