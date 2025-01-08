import os
import zipfile
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Unggah Gambar", "Riwayat"])

# Memuat model yang sudah dilatih
model_path = 'disease.h5'  # Specify your model file path
if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di {model_path}")
else:
    model = load_model(model_path, compile=False)

    # Memuat nama kelas dari model
    classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']  # Update these classes to match your dataset

    # Fungsi untuk memproses gambar input
    def preprocess_image(img):
        img = img.resize((180, 180))  # Adjust image size as required
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    # Fungsi untuk memprediksi gambar
    def predict_image(img_array):
        preds = model.predict(img_array)  # Predict with the model
        class_idx = np.argmax(preds, axis=1)  # Get class with the highest probability
        return classes[class_idx[0]], preds[0][class_idx[0]]  # Return class and confidence

    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header dengan gambar dan deskripsi
    st.image("flower_classification_logo.png", width=150)  # Replace with your logo
    st.title("Klasifikasi Jenis Bunga")

    if menu == "Beranda":
        st.markdown("""
        Aplikasi klasifikasi jenis bunga ini menggunakan kecerdasan buatan (AI) untuk membantu mengidentifikasi jenis bunga
        berdasarkan gambar yang diunggah oleh pengguna. Dengan aplikasi ini, Anda dapat mengupload gambar bunga atau mengambil
        gambar langsung menggunakan kamera untuk mengetahui jenis bunga tersebut.
        """, unsafe_allow_html=True)

    elif menu == "Kamera":
        # Menampilkan pilihan untuk mengambil gambar menggunakan kamera
        camera_input = st.camera_input("Ambil gambar untuk diprediksi")

        if camera_input is not None:
            # Menampilkan gambar yang diambil
            st.image(camera_input, caption="Gambar yang diambil.", use_container_width=True)

            # Memproses gambar
            try:
                img = Image.open(camera_input)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.stop()  # Berhenti jika ada kesalahan dalam memproses gambar

            img_array = preprocess_image(img)

            # Prediksi
            label, confidence = predict_image(img_array)
            st.write(f"Prediksi: {label}")
            st.write(f"Probabilitas: {confidence:.2f}")

            # Menyimpan gambar dan hasil prediksi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append({
                "image": img_bytes,
                "label": label,
                "confidence": confidence
            })

    elif menu == "Unggah Gambar":
        # Menampilkan fitur unggah gambar
        uploaded_file = st.file_uploader("Pilih gambar untuk diprediksi", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Menampilkan gambar yang diunggah
            st.image(uploaded_file, caption="Gambar yang diunggah.", use_container_width=True)

            # Memproses gambar
            try:
                img = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
                st.stop()  # Berhenti jika ada kesalahan dalam memproses gambar

            img_array = preprocess_image(img)

            # Prediksi
            label, confidence = predict_image(img_array)
            st.write(f"Prediksi: {label}")
            st.write(f"Probabilitas: {confidence:.2f}")

            # Menyimpan gambar dan hasil prediksi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append({
                "image": img_bytes,
                "label": label,
                "confidence": confidence
            })

    elif menu == "Riwayat":
        # Menampilkan riwayat hasil prediksi
        if len(st.session_state.history) == 0:
            st.write("Tidak ada riwayat prediksi.")
        else:
            st.write("Riwayat Prediksi Bunga:")

            # Loop untuk menampilkan setiap entri dalam riwayat
            for i, entry in enumerate(st.session_state.history):
                # Menampilkan gambar dari riwayat
                st.image(entry["image"], caption=f"Prediksi {i+1}: {entry['label']} (Probabilitas: {entry['confidence']:.2f})", use_container_width=True)
                st.write(f"**Prediksi**: {entry['label']}")
                st.write(f"**Probabilitas**: {entry['confidence']:.2f}")

                # Menambahkan tombol hapus
                if st.button(f"Hapus Prediksi {i+1}", key=f"hapus_{i}"):
                    # Menghapus entri dari riwayat
                    st.session_state.history.pop(i)
                
                st.markdown("---")

# Menambahkan CSS kustom untuk mempercantik tampilan
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #6495ED;
            color: white;
            padding: 20px 0;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        }
        .css-ffhzg2 {
            font-size: 1.25em;
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stImage>img {
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)
