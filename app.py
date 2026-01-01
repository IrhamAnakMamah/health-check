import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HealthCheck - Deteksi Risiko Obesitas",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Load file yang baru diexport (Random Forest & OHE)
        model = joblib.load('model/rf_model.joblib')
        scaler = joblib.load('model/scaler.joblib')
        model_cols = joblib.load('model/model_columns.joblib')
        le_dict = joblib.load('model/label_encoder.joblib')
        ohe_gender = joblib.load('model/ohe_gender.joblib') # Load OHE Gender
        return model, scaler, model_cols, le_dict, ohe_gender
    except FileNotFoundError as e:
        st.error(f"⚠️ File asset tidak lengkap: {e}. Pastikan rf_model.joblib, ohe_gender.joblib, dll ada di folder yang sama.")
        return None, None, None, None, None

model, scaler, model_cols, le_dict, ohe_gender = load_assets()

# --- 3. HEADER ---
st.title("⚖️ HealthCheck")
st.markdown("### Deteksi Dini Risiko Obesitas & Saran Kesehatan")
st.markdown("---")

# --- 4. INPUT USER (MENGGUNAKAN TABS) ---
if model is not None:
    # Inisialisasi session state untuk menyimpan input antar tab
    if 'input_data' not in st.session_state:
        st.session_state['input_data'] = {}

    tab1, tab2, tab3 = st.tabs(["👤 Data Fisik", "🍽️ Pola Makan", "🏃 Aktivitas"])

    with tab1:
        st.header("Profil Fisik")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"], format_func=lambda x: "Laki-laki" if x == "Male" else "Perempuan")
            age = st.number_input("Usia (Tahun)", 10, 100, 20)
        with col2:
            height = st.number_input("Tinggi Badan (cm)", 100, 250, 170)
            weight = st.number_input("Berat Badan (kg)", 30, 200, 70)
        
        st.info("💡 Tinggi dalam cm akan otomatis dikonversi ke meter oleh sistem.")

    with tab2:
        st.header("Kebiasaan Makan")
        
        # Pertanyaan yang lebih manusiawi
        favc = st.radio("Sering makan makanan tinggi kalori? (Gorengan, fast food, manis)", ['no', 'yes'], 
                        format_func=lambda x: "Ya, Sering" if x == 'yes' else "Jarang/Tidak")
        
        fcvc = st.select_slider("Seberapa sering makan sayur?", 
                                options=[1.0, 2.0, 3.0],
                                format_func=lambda x: {1.0: "Tidak Pernah", 2.0: "Kadang-kadang", 3.0: "Setiap Makan"}[x])
        
        ncp = st.slider("Berapa kali makan besar (nasi/berat) sehari?", 1, 4, 3)
        
        caec = st.selectbox("Suka ngemil di luar jam makan?", ['no', 'Sometimes', 'Frequently', 'Always'],
                            format_func=lambda x: {
                                'no': 'Tidak Pernah', 'Sometimes': 'Kadang-kadang', 
                                'Frequently': 'Sering', 'Always': 'Setiap Saat'}[x])

    with tab3:
        st.header("Gaya Hidup")
        faf = st.select_slider("Seberapa sering olahraga fisik?", 
                               options=[0.0, 1.0, 2.0, 3.0],
                               format_func=lambda x: {
                                   0.0: "Tidak Pernah", 1.0: "1-2 hari/minggu", 
                                   2.0: "3-4 hari/minggu", 3.0: "Hampir tiap hari"}[x])
        
        st.markdown("---")
        predict_btn = st.button("🔍 Analisis Kesehatan Saya")

    # --- 5. LOGIKA PREDIKSI ---
    if predict_btn:
        with st.spinner('Sedang menganalisis data tubuh Anda...'):
            time.sleep(1) # Efek loading
            
            # Konversi Tinggi cm ke Meter (karena model butuh Meter)
            height_m = height / 100.0

            # Siapkan Data Input
            # Note: Gunakan nama kolom asli sebelum encoded
            raw_input = {
                'Age': age,
                'Height': height_m,
                'Weight': weight,
                'FAVC': favc,
                'FCVC': fcvc,
                'NCP': float(ncp),
                'CAEC': caec,
                'FAF': faf,
                'Gender': gender
            }
            
            df_input = pd.DataFrame([raw_input])

            # --- PREPROCESSING ---
            try:
                # 1. Label Encoding (FAVC & CAEC)
                # Pastikan input user ada di dalam classes label encoder
                df_input['FAVC'] = le_dict['FAVC'].transform([df_input['FAVC'][0]])[0]
                df_input['CAEC'] = le_dict['CAEC'].transform([df_input['CAEC'][0]])[0]

                # 2. Scaling Numerik (Robust Scaler)
                numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF']
                for col in numeric_cols:
                    if col in scaler:
                        df_input[col] = scaler[col].transform(df_input[[col]])

                # 3. One Hot Encoding Gender (Menggunakan Artifact ohe_gender)
                # Ini lebih aman daripada if/else manual karena mengikuti struktur saat training
                gender_encoded = ohe_gender.transform(df_input[['Gender']])
                gender_encoded_df = pd.DataFrame(
                    gender_encoded, 
                    columns=ohe_gender.get_feature_names_out(['Gender'])
                )
                
                # Gabungkan dan drop kolom Gender asli
                df_input = pd.concat([df_input.drop(columns=['Gender']), gender_encoded_df], axis=1)

                # 4. Reorder Columns (Wajib sama persis dengan X_train saat training)
                # Membuat dataframe baru dengan urutan kolom yang benar
                df_final = pd.DataFrame(columns=model_cols)
                for col in model_cols:
                    # Isi nilai jika kolom ada di df_input, jika tidak isi 0
                    if col in df_input.columns:
                        df_final.loc[0, col] = df_input.iloc[0][col]
                    else:
                        df_final.loc[0, col] = 0

                # --- PREDIKSI ---
                pred_idx = model.predict(df_final)[0]
                class_names = le_dict['NObeyesdad'].classes_
                result_label = class_names[pred_idx]

                # --- TAMPILKAN HASIL ---
                st.markdown("---")
                
                # Hitung BMI Manual untuk referensi user
                bmi = weight / (height_m ** 2)
                
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    st.markdown("### 📊 BMI Anda")
                    st.metric(label="Body Mass Index", value=f"{bmi:.1f}")
                    if bmi < 18.5:
                        st.caption("Kategori: Kurus")
                    elif 18.5 <= bmi < 24.9:
                        st.caption("Kategori: Normal")
                    elif 25 <= bmi < 29.9:
                        st.caption("Kategori: Overweight")
                    else:
                        st.caption("Kategori: Obesitas")

                with col_res2:
                    st.markdown("### 🤖 Hasil Analisis AI")
                    
                    # Logic Warna & Pesan
                    if "Normal" in result_label:
                        st.success(f"**Kategori: {result_label}**")
                        st.write("🎉 **Bagus!** Berat badan Anda ideal. Pertahankan pola makan sehat dan olahraga teratur ini.")
                    elif "Insufficient" in result_label:
                        st.warning(f"**Kategori: {result_label}**")
                        st.write("⚠️ **Berat Badan Kurang.** Disarankan meningkatkan asupan kalori bergizi (protein & karbohidrat kompleks).")
                    elif "Overweight" in result_label:
                        st.warning(f"**Kategori: {result_label}**")
                        st.write("⚠️ **Kelebihan Berat Badan.** Mulai kurangi gula/manis dan usahakan jalan kaki minimal 30 menit sehari.")
                    else: # Obesity Types
                        st.error(f"**Kategori: {result_label}**")
                        st.write("🚨 **Risiko Obesitas Terdeteksi.**")
                        st.markdown("""
                        **Saran Aksi:**
                        1. 🛑 Kurangi makanan 'FAVC' (Gorengan/Manis/Tepung) secara signifikan.
                        2. 🥗 Tingkatkan porsi sayur di setiap kali makan.
                        3. 🩺 Sangat disarankan berkonsultasi dengan ahli gizi atau dokter.
                        """)
                
                # Menampilkan Probabilitas (Fitur Random Forest)
                with st.expander("Lihat Detail Probabilitas (Analisis Mendalam)"):
                    try:
                        proba = model.predict_proba(df_final)[0]
                        probs_df = pd.DataFrame({
                            'Kategori': class_names,
                            'Probabilitas': proba
                        })
                        # Mengurutkan biar lebih enak dilihat
                        probs_df = probs_df.sort_values(by='Probabilitas', ascending=False)
                        
                        st.dataframe(probs_df.style.format({'Probabilitas': '{:.2%}'}), use_container_width=True)
                        st.bar_chart(probs_df.set_index('Kategori'))
                    except:
                        st.write("Model tidak mendukung probabilitas.")

            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")