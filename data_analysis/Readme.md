# BDD Dataset Analysis (Dockerized)

This project provides a **Dockerized Python 3.11 environment** with all necessary dependencies to facilitate to run **Data Visualization** and **Data Statistics Dashboard** of the BDD100K dataset.

---

## Setup Guide

### 1. Download the Dataset

To begin, download the **BDD100K dataset (images & labels)** from the official source:

- **Images**: [BDD100K Official Website](https://bdd-data.berkeley.edu/)
- **Labels**: [Download Labels](https://bdd-data.berkeley.edu/)

Once downloaded, extract the dataset and place in data folder of the same repo for mounting later.

### 2. Navigate to the Project Directory

Ensure you are inside the correct project folder before running Docker commands:

```bash
cd /path/to/data-analysis
```

### 3. Build the Docker Image
Use the following command to create a Docker image for the analysis environment:

```bash
docker build -t bdd-analysis 
```

### 4. Run the Container with Dataset Access
Start the container and mount the dataset so it can be accessed within the environment:

```bash
docker run -it -p 8888:8888 -v $(pwd):/workspace -v /path/to/bdd_dataset:/workspace/data bdd-analysis
```

### 5. Process
1. Running docker included analyzing data and visualizing groundtruths, unique and rare images, occluded and truncated images and saving the visualizations to the portal.
2. Creates Streamlit dashboard with data statistics.

### Notes:
Ensure Docker is installed and running before executing the commands.

Modify the dataset path (/path/to/bdd_dataset) based on your local system.

Now you're all set to explore and analyze the BDD100K dataset!