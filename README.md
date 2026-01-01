# ⚖️ HealthCheck - Deteksi Risiko Obesitas

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**HealthCheck** adalah aplikasi web interaktif yang memanfaatkan *Machine Learning* untuk mendeteksi tingkat risiko obesitas seseorang. Aplikasi ini dirancang agar mudah digunakan oleh orang awam untuk memahami profil kesehatan mereka berdasarkan kebiasaan makan dan kondisi fisik.

Dibangun dengan **Streamlit** dan ditenagai oleh model **Random Forest Classifier** yang memiliki akurasi tinggi (~94%), aplikasi ini tidak hanya memberikan klasifikasi berat badan tetapi juga saran kesehatan yang dipersonalisasi.

---

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Teknologi](#-teknologi)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Detail Model](#-detail-model)
- [Dataset](#-dataset)
- [Kontribusi](#-kontribusi)

---

## ✨ Fitur Utama

* **Klasifikasi 7 Tingkat**: Memprediksi kategori dari *Insufficient Weight* (Kurang Berat Badan) hingga *Obesity Type III* (Obesitas Tipe 3).
* **Input User-Friendly**: Formulir interaktif untuk memasukkan data fisik (tinggi, berat), pola makan (frekuensi makan, konsumsi sayur, air), dan gaya hidup (olahraga, penggunaan gadget).
* **Kalkulator BMI Otomatis**: Menghitung indeks massa tubuh secara real-time sebagai referensi awal.
* **Saran Personal**: Memberikan rekomendasi kesehatan yang spesifik berdasarkan hasil prediksi (misal: saran diet untuk *Overweight* vs saran medis untuk *Obesity*).
* **Visualisasi Probabilitas**: Menampilkan grafik tingkat keyakinan model terhadap hasil prediksi untuk transparansi analisis.

---

## 🛠 Teknologi

Project ini dibangun menggunakan:

* **Python**: Bahasa pemrograman utama.
* **Streamlit**: Framework untuk membuat antarmuka web data science.
* **Scikit-Learn**: Library machine learning untuk pelatihan model (Random Forest).
* **Pandas & NumPy**: Untuk manipulasi dan analisis data.
* **Joblib**: Untuk menyimpan dan memuat model yang telah dilatih.
* **Matplotlib/Seaborn**: (Digunakan dalam notebook) Untuk visualisasi data saat pengembangan.

---

## 💻 Instalasi

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer lokal Anda:

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/irhamanakmamah/health-check.git](https://github.com/irhamanakmamah/health-check.git)
    cd health-check
    ```

2.  **Buat Virtual Environment (Disarankan)**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependensi**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```

Aplikasi akan otomatis terbuka di browser default Anda di alamat `http://localhost:8501`.

---

## 📱 Cara Penggunaan

1.  Buka aplikasi di browser.
2.  Isi **Data Fisik** (Gender, Umur, Tinggi, Berat).
3.  Lengkapi tab **Pola Makan** (Jumlah makan, konsumsi air, ngemil, dll).
4.  Lengkapi tab **Gaya Hidup** (Olahraga, waktu penggunaan teknologi, transportasi).
5.  Klik tombol **"🔍 Analisis Risiko Kesehatan"**.
6.  Lihat hasil prediksi kategori obesitas dan baca saran kesehatan yang diberikan.

---

## 📂 Struktur Proyek

```text
health-check/
├── app.py                  # File utama aplikasi Streamlit
├── README.md               # Dokumentasi proyek
├── requirements.txt        # Daftar library
├── build/                  # Folder pengembangan model
│   ├── ObesityDataSet...   # Dataset mentah (.csv)
│   └── build_model.ipynb   # Notebook eksperimen & pelatihan
└── model/                  # Artefak model (JANGAN DIHAPUS)
    ├── rf_model.joblib     # Model Random Forest
    ├── scaler.joblib       # RobustScaler
    ├── label_encoder.joblib # Label Encoder
    ├── ohe_gender.joblib   # OneHotEncoder
    └── model_columns.joblib # Metadata kolom
```

## 📊 Detail Model

Model dilatih menggunakan algoritma **Random Forest Classifier** yang dipilih karena kemampuannya menangani data non-linear dan memberikan akurasi tinggi.

* **Preprocessing**:
    * `RobustScaler` untuk fitur numerik (menangani outlier).
    * `OneHotEncoder` untuk fitur Gender.
    * `LabelEncoder` untuk fitur kategorikal ordinal.
* **Performa**: Mencapai akurasi **~94%** pada data uji, dengan presisi dan recall yang seimbang di seluruh kelas.

---

## 📚 Dataset

Dataset yang digunakan bersumber dari **UCI Machine Learning Repository**:
* **Nama**: *Estimation of obesity levels based on eating habits and physical condition*.
* **Asal Data**: Meksiko, Peru, dan Kolombia.
* **Atribut**: 17 atribut mencakup data fisik, kebiasaan makan, dan aktivitas fisik.

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan fork repositori ini dan buat Pull Request untuk fitur baru, perbaikan bug, atau peningkatan dokumentasi.

1.  Fork Project
2.  Buat Feature Branch (`git checkout -b feature/"yang mau ditambahkan"`)
3.  Commit Perubahan (`git commit -m 'Add some "yang mau ditambahkan"'`)
4.  Push ke Branch (`git push origin feature/"yang mau ditambahkan"`)
5.  Buka Pull Request

---
