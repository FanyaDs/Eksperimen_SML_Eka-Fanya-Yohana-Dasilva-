# ================================================================
# AUTOMATE_EKA-FANYA-YOHANA-DASILVA.PY
# Otomatisasi Data Preprocessing - Proyek Analisis Sentimen Wisata Bali
# ================================================================

import pandas as pd
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------------------------------------
# Instalasi dan inisialisasi resource (hanya jika belum ada)
# ------------------------------------------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # fix error LookupError: punkt_tab not found

# ------------------------------------------------------------
# Fungsi utama preprocessing teks
# ------------------------------------------------------------
def clean_text(teks):
    # 1. Case folding
    teks = teks.lower()

    # 2. Hapus URL, angka, dan tanda baca
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = teks.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenizing
    tokens = word_tokenize(teks)

    # 4. Stopword removal
    stop_words = set(stopwords.words('indonesian'))
    tokens = [kata for kata in tokens if kata not in stop_words]

    # 5. Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    hasil = [stemmer.stem(kata) for kata in tokens]

    # 6. Gabungkan kembali jadi teks bersih
    return " ".join(hasil)

# ------------------------------------------------------------
# Fungsi utama pipeline preprocessing
# ------------------------------------------------------------
def run_preprocessing(input_path, output_path):
    print("=== MULAI PROSES PREPROCESSING DATASET WISATA BALI ===")

    # Membaca dataset mentah
    df = pd.read_csv(input_path)
    print(f"Dataset berhasil dibaca: {len(df)} baris")

    # Terapkan fungsi pembersihan teks
    df['clean_text'] = df['text'].apply(clean_text)

    # Hapus duplikat jika ada
    before = len(df)
    df.drop_duplicates(subset='clean_text', inplace=True)
    after = len(df)
    print(f"Data duplikat yang dihapus: {before - after}")

    # Simpan hasil preprocessing
    df.to_csv(output_path, index=False)
    print(f"✅ File hasil preprocessing disimpan sebagai: {output_path}")
    print("=== SELESAI ===")

# ------------------------------------------------------------
# Jalankan otomatisasi
# ------------------------------------------------------------
if __name__ == "__main__":
    input_path = "namadataset_raw/wisata_bali_labeled.csv"
    output_path = "preprocessing/namadataset_preprocessing/wisata_bali_preprocessed.csv"
    run_preprocessing(input_path, output_path)

