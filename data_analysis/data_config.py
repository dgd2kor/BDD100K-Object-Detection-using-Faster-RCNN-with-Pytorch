class Config:
    """Configuration file for dataset paths and hyperparameters."""
    train_annotations = "/app/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    val_annotations = "/app/data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    train_image_path = "/app/data/bdd100k_images_100k/bdd100k/images/100k/train"
    val_image_path = "/app/data/bdd100k_images_100k/bdd100k/images/100k/val"

    num_visuals = 5
    num_occ_trun_visuals = 5
    num_unique_visuals = 5
    visuals_output_path = "./visuals"
