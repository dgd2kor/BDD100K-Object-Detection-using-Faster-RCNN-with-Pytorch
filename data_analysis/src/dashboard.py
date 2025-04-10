import streamlit as st
import plotly.express as px
from data_config import Config


st.title("BDD100K Dataset Dashboard")

from analyze_dataset import category_count, distribution_analysis
from parse_json import load_annotations
from visualize_data import visualize_all_groundtruths, visualize_unique_images, visualize_occluded_truncated

config = Config()

train_annotations_path = config.train_annotations
val_annotations_path = config.val_annotations

train_count_df = category_count(train_annotations_path)
val_count_df = category_count(val_annotations_path)

data_distribution_df = distribution_analysis(train_count_df, val_count_df)

# Save visualizations
train_annotations = load_annotations(config.train_annotations)

visualize_all_groundtruths(config.train_image_path, train_annotations, config.num_visuals)

visualize_unique_images(config.train_image_path, train_annotations, config.num_unique_visuals)

visualize_occluded_truncated(config.train_image_path, train_annotations, config.num_occ_trun_visuals)


# Bar Chart: Train vs Val Distribution
fig_bar = px.bar(data_distribution_df, x="Category", y=["train_count", "val_count"], title="Train vs Val Object Distribution", barmode="group")
st.plotly_chart(fig_bar)

# Pie Chart: Train Data Distribution
fig_pie_train = px.pie(data_distribution_df, names="Category", values="train_count", title="Train Set Distribution")
st.plotly_chart(fig_pie_train)

# Pie Chart: Val Data Distribution
fig_pie_val = px.pie(data_distribution_df, names="Category", values="val_count", title="Validation Set Distribution")
st.plotly_chart(fig_pie_val)
