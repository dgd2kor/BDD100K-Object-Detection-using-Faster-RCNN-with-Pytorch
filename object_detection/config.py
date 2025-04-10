class Config:
    """Configuration file for dataset paths and hyperparameters."""
    train_annotations = "data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    val_annotations = "data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    train_image_path = "data/bdd100k_images_100k/bdd100k/images/100k/train"
    val_image_path = "data/bdd100k_images_100k/bdd100k/images/100k/val"

    num_visuals = 5
    subset_train = True
    num_classes = 12
    batch_size = 16 
    lr = 1e-4
    epochs = 1

    checkpoint_path = "object_detection/weights/bdd100k_24.pth"
    class_to_idx = {
        "bus": 0, "traffic light": 1, "traffic sign": 2,
        "person": 3, "bike": 4, "truck": 5,
        "motor": 6, "car": 7, "train": 8, "rider": 9
    }
    predictions_path = "outputs/predictions"
    
    metrics_path = "outputs/metrics"
    
    pred_visuals_path = "outputs/pred_visualizations"

    iou_threshold = 0.5