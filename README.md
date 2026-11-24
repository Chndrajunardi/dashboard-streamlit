# Dashboard Klasifikasi Subtipe Molekuler Kanker Payudara

**Dashboard Klasifikasi Subtipe Molekuler Kanker Payudara Berbasis Machine Learning Menggunakan Data TCGA-BRCA**

## Deskripsi

Repository ini berisi source code pengembangan aplikasi dashboard berbasis **Streamlit** untuk klasifikasi subtipe molekuler kanker payudara menggunakan data ekspresi gen The Cancer Genome Atlas (TCGA-BRCA).

Aplikasi ini merupakan implementasi dari penelitian Machine Learning yang mengembangkan model klasifikasi menggunakan algoritma **Support Vector Machine (SVM)** dan **Multi-Layer Perceptron (MLP)** dengan sistem klasifikasi lima subtipe molekuler PAM50.

Dashboard dirancang untuk menyediakan antarmuka interaktif yang memungkinkan pengguna memasukkan data ekspresi gen, menjalankan proses klasifikasi, serta melihat hasil prediksi subtipe molekuler kanker payudara.

## Tujuan Pengembangan

- Mengimplementasikan model Machine Learning ke dalam aplikasi berbasis web.
- Menyediakan antarmuka interaktif untuk klasifikasi subtipe molekuler kanker payudara.
- Menampilkan hasil prediksi berdasarkan data ekspresi gen.
- Mempermudah pemahaman hasil klasifikasi melalui visualisasi dashboard.
- Mendukung demonstrasi penerapan Machine Learning dalam analisis ekspresi gen TCGA-BRCA.

## Subtipe Molekuler PAM50

Dashboard menggunakan lima kategori subtipe molekuler PAM50:

| Subtipe | Keterangan |
|---|---|
| Basal-like | Subtipe molekuler Basal-like |
| HER2-enriched | Subtipe molekuler HER2-enriched |
| Luminal A | Subtipe molekuler Luminal A |
| Luminal B | Subtipe molekuler Luminal B |
| Normal-like | Subtipe molekuler Normal-like |

## Fitur Dashboard

Dashboard dikembangkan untuk menyediakan fitur-fitur berikut:

- **Informasi Penelitian:** Penjelasan mengenai latar belakang, tujuan, dan metode penelitian.
- **Alur Penggunaan:** Panduan penggunaan aplikasi.
- **Informasi Subtipe PAM50:** Penjelasan mengenai lima subtipe molekuler kanker payudara.
- **Klasifikasi:** Proses klasifikasi berdasarkan data ekspresi gen menggunakan model yang tersedia.
- **Hasil Prediksi:** Menampilkan hasil prediksi subtipe molekuler.
- **Visualisasi Hasil:** Menampilkan informasi hasil klasifikasi dalam bentuk visualisasi interaktif sesuai implementasi aplikasi.

## Struktur Repository

```text
dashboard-streamlit/
│
├── final_model/
│   └── Model dan artefak preprocessing
│
├── app.py
│   └── Source code utama aplikasi Streamlit
│
└── README.md
    └── Dokumentasi repository
```

## Model dan Artefak Pendukung

Folder `final_model/` digunakan untuk menyimpan model Machine Learning dan artefak preprocessing yang diperlukan oleh aplikasi.

Artefak model MLP yang digunakan dalam penelitian meliputi:

| File | Keterangan |
|---|---|
| `tuned_mlp_final.keras` | Model MLP hasil pelatihan dan optimasi |
| `standard_scaler.joblib` | Objek standardisasi fitur |
| `selected_gene_names.json` | Daftar dan urutan gen terpilih |
| `label_mapping.json` | Pemetaan label kelas PAM50 |

Model dan artefak preprocessing digunakan untuk memastikan proses prediksi pada dashboard mengikuti konfigurasi yang sama dengan proses pelatihan.

## Teknologi yang Digunakan

- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Joblib

## Instalasi dan Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone <URL-REPOSITORY>
```

### 2. Masuk ke Direktori Repository

```bash
cd dashboard-streamlit
```

### 3. Instalasi Library

```bash
pip install streamlit tensorflow scikit-learn numpy pandas matplotlib joblib
```

### 4. Menjalankan Dashboard

```bash
streamlit run app.py
```

Setelah aplikasi dijalankan, Streamlit akan menampilkan alamat lokal untuk mengakses dashboard melalui browser.

## Catatan Penggunaan

1. Pastikan seluruh library yang diperlukan telah terinstal.
2. Pastikan model dan artefak preprocessing tersedia di dalam folder `final_model/`.
3. Gunakan data ekspresi gen dengan format yang sesuai.
4. Pastikan nama dan urutan gen sesuai dengan fitur yang digunakan pada proses pelatihan.
5. Jalankan aplikasi menggunakan perintah Streamlit.
6. Interpretasikan hasil prediksi sebagai keluaran model Machine Learning, bukan sebagai diagnosis klinis.

## Keterangan Penelitian

Dashboard ini merupakan bagian dari penelitian skripsi dengan judul:

**Dashboard Klasifikasi Subtipe Molekuler Kanker Payudara Berbasis Machine Learning Menggunakan Data TCGA-BRCA**

Aplikasi dikembangkan sebagai bentuk implementasi model Machine Learning untuk mengklasifikasikan subtipe molekuler kanker payudara berdasarkan pola ekspresi gen.

**Disclaimer:** Aplikasi ini dikembangkan untuk kepentingan penelitian dan edukasi akademik, bukan sebagai alat diagnosis klinis atau pengganti keputusan tenaga medis.
