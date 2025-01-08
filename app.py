import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# Set up the Streamlit page configuration
st.set_page_config(page_title="Flower Classification", layout="centered")

# Title of the app
st.title("Flower Classification App")

# Path for the dataset (if using Google Drive)
base_dir = '/content/drive/MyDrive/flowers/'

# Function to upload and display sample images
def show_sample_images(directory):
    try:
        dirs = os.listdir(directory)
        count = 0
        for dir in dirs:
            dir_path = os.path.join(directory, dir)
            if os.path.isdir(dir_path):  # Ensure it's a directory, not a file
                files = os.listdir(dir_path)
                count += len(files)
        st.write(f'Total images in the flower dataset: {count}')

        flower_names = [d for d in dirs if os.path.isdir(os.path.join(directory, d))]  # Filter directories
        st.write(f"Flower categories: {flower_names}")

        # Show some sample images
        st.write("### Sample Images from Dataset")
        sample_images = []
        for dir in flower_names:
            dir_path = os.path.join(directory, dir)
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))][:3]  # Only select files
            sample_images.extend([os.path.join(dir_path, file) for file in files])

        # Display sample images
        cols = st.columns(3)
        for idx, image_path in enumerate(sample_images):
            with cols[idx % 3]:
                img = load_img(image_path, target_size=(180, 180))
                st.image(img, caption=flower_names[idx % len(flower_names)])
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")

# Handle file upload and extraction
if os.path.exists(base_dir):
    st.write("Dataset found on Google Drive.")
    show_sample_images(base_dir)
else:
    st.write("Dataset not found in Google Drive.")
    uploaded_folder = st.file_uploader("Upload a folder of flower images (zip file)", type=["zip"])

    if uploaded_folder is not None:
        zip_path = "flowers.zip"  # Temporary location
        with open(zip_path, "wb") as f:
            f.write(uploaded_folder.getbuffer())

        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        st.write(f"Dataset extracted to {base_dir}")
        show_sample_images(base_dir)

# Load dataset
img_size = 180
batch = 32
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    seed=123,
    validation_split=0.2,
    subset='training',
    batch_size=batch,
    image_size=(img_size, img_size)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    seed=123,
    validation_split=0.2,
    subset='validation',
    batch_size=batch,
    image_size=(img_size, img_size)
)

flower_names = train_ds.class_names

# Data Preprocessing and Augmentation
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Define the model architecture
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5)  # Number of classes
])

model.summary()

# Compile the model before training
model.compile(
    optimizer='adam',  # Optimizer choice, 'adam' is a good default
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # For multi-class classification
    metrics=['accuracy']  # Metric to track during training
)

# Train the
