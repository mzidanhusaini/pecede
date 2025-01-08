import os

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

# Load the flower dataset directory
base_dir = '/content/drive/MyDrive/flowers/'

# Create a simple function to display some sample images from the dataset
def show_sample_images():
    dirs = os.listdir(base_dir)
    count = 0
    for dir in dirs:
        files = list(os.listdir(base_dir + dir))
        count += len(files)
    st.write(f'Total images in the flower dataset: {count}')
    
    flower_names = os.listdir(base_dir)  # Assuming flower directories match the flower names
    st.write(f"Flower categories: {flower_names}")

    # Show some sample images
    st.write("### Sample Images from Dataset")
    sample_images = []
    for dir in flower_names:
        dir_path = os.path.join(base_dir, dir)
        files = list(os.listdir(dir_path))[:3]  # Take 3 images from each category
        sample_images.extend([os.path.join(dir_path, file) for file in files])
    
    cols = st.columns(3)
    for idx, image_path in enumerate(sample_images):
        with cols[idx % 3]:
            img = load_img(image_path, target_size=(180, 180))
            st.image(img, caption=flower_names[idx % len(flower_names)])

# Show sample images
show_sample_images()

# Load the dataset
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

# Train the model
history = model.fit(train_ds, epochs=15, validation_data=val_ds)

# Function to classify uploaded images
def classify_images(image):
    input_image = tf.keras.utils.load_img(image, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result)*100:.2f}%"
    return outcome

# Streamlit interface for user input
st.write("### Upload an Image for Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Call the classify function and display result
    outcome = classify_images(uploaded_file)
    st.write(outcome)
