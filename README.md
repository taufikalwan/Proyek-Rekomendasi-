# Laporan Proyek Sistem Rekomendasi - TAUFIK ALWAN

Di era digital saat ini, teknologi *machine learning* menjadi salah satu pilar utama dalam proses pengolahan dan analisis data. Teknologi ini mampu memberikan solusi efektif dalam berbagai permasalahan, termasuk dalam sistem rekomendasi. Sistem rekomendasi bertujuan untuk memberikan saran atau pilihan yang paling sesuai bagi pengguna, berdasarkan preferensi atau riwayat interaksi mereka sebelumnya.

Teknologi rekomendasi telah banyak diimplementasikan di berbagai platform, seperti media sosial yang menyajikan konten sesuai minat pengguna, maupun marketplace yang menawarkan produk berdasarkan preferensi pembeli. Perusahaan besar seperti Amazon, Netflix, dan Spotify juga telah mengadopsi sistem rekomendasi dalam skala global untuk meningkatkan pengalaman pengguna mereka.

Proyek ini bertujuan untuk mengembangkan model sistem rekomendasi lagu yang mampu memberikan saran musik yang relevan kepada pengguna. Pengembangan model ini berfokus pada dua pendekatan utama dalam *machine learning*, yaitu *Content-Based Filtering* dan *Collaborative Filtering*. Proyek ini dianggap penting karena dapat membantu pengguna dalam memilih lagu berdasarkan kebiasaan mendengarkan mereka sebelumnya, sekaligus membandingkan efektivitas kedua pendekatan dalam menghasilkan rekomendasi yang paling akurat.

Dengan kehadiran sistem rekomendasi yang tepat dan efisien, pengguna akan lebih mudah dalam mengambil keputusan untuk memilih lagu yang ingin mereka dengarkan, sehingga pengalaman penggunaan aplikasi menjadi lebih personal dan menyenangkan. Penelitian ini juga merujuk pada beberapa studi sebelumnya yang relevan, antara lain:

* *Menakar Preferensi Musik di Kalangan Remaja: Antara Musik Populer dan Musik Klasik*, yang menyoroti kecenderungan remaja dalam memilih genre musik tertentu.
* *Pengembangan Sistem Rekomendasi Berbasis Kecerdasan Buatan untuk Meningkatkan Pengalaman Pengguna di Platform E-Commerce*, yang menekankan pentingnya penerapan AI dalam meningkatkan kenyamanan pengguna di e-commerce.
* *Pemanfaatan Sistem Rekomendasi Menggunakan Content-Based Filtering pada Hotel di Palangka Raya*, yang menunjukkan efektivitas pendekatan Content-Based dalam menyajikan rekomendasi akurat.
* *Preferensi Musik di Kalangan Remaja*, yang menjelaskan bahwa berbagai faktor turut memengaruhi ketertarikan seseorang terhadap genre musik tertentu.

---

---

## ⚠️ Rumusan Masalah

Penelitian ini dilatarbelakangi oleh beberapa permasalahan utama yang ingin diselesaikan, yaitu:

1. **Bagaimana sistem rekomendasi dapat memberikan saran lagu yang relevan berdasarkan kemiripan konten, seperti judul lagu, dan sejauh mana relevansi tersebut tercermin melalui hasil evaluasi model?**

2. **Bagaimana sistem dapat memberikan rekomendasi lagu secara personal kepada pengguna berdasarkan preferensi artis favorit sebelumnya, serta seberapa akurat model dalam memprediksi penilaian terhadap lagu yang belum pernah didengarkan oleh pengguna tersebut?**

---

---

## ✨ Tujuan Penelitian

Berdasarkan rumusan masalah yang telah dikemukakan, tujuan dari penelitian ini adalah sebagai berikut:

1. **Menentukan model sistem rekomendasi yang efektif dalam memberikan saran lagu yang relevan kepada pengguna.**

2. **Melakukan perbandingan antara beberapa pendekatan sistem rekomendasi untuk memperoleh solusi terbaik dalam konteks rekomendasi lagu yang personal dan akurat.**

---

---

## 💡 Pendekatan Solusi

Untuk menjawab permasalahan yang telah diidentifikasi, penelitian ini menggunakan pendekatan berbasis *machine learning* dalam membangun sistem rekomendasi lagu. Dua metode utama yang digunakan adalah **Content-Based Filtering** dan **Collaborative Filtering**.

Pada pendekatan *Collaborative Filtering*, penelitian ini memanfaatkan teknik *neural network* guna meningkatkan kemampuan model dalam memahami preferensi pengguna berdasarkan interaksi sebelumnya. Sementara itu, *Content-Based Filtering* digunakan untuk merekomendasikan lagu berdasarkan kemiripan konten, seperti kemiripan judul lagu atau atribut lainnya.

Proses pengembangan sistem dilakukan melalui beberapa tahapan penting, meliputi:

* Pengumpulan dan pengolahan data,
* Eksplorasi dan ekstraksi fitur,
* Pemilihan serta penerapan algoritma rekomendasi, dan
* Evaluasi performa model untuk memastikan tingkat akurasi dan relevansi rekomendasi yang optimal.

Pendekatan ini diharapkan mampu memberikan rekomendasi lagu yang relevan dan personal bagi setiap pengguna.

---

---

## 📑 Pemahaman Data (*Data Understanding*)

Dataset yang digunakan dalam penelitian ini merupakan kumpulan data lagu yang terorganisir dengan baik dan diperoleh dari platform **Kaggle** dengan nama **"Song Dataset"**. Dataset ini berisi informasi penting mengenai lagu-lagu yang dapat digunakan untuk membangun sistem rekomendasi.

Adapun variabel-variabel yang terdapat dalam dataset tersebut antara lain:

* **Name of the Song**: Berisi nama atau judul dari masing-masing lagu.
* **Artist**: Menunjukkan nama artis atau penyanyi dari lagu tersebut.
* **Date of Release**: Mencatat tanggal resmi perilisan lagu.
* **Description**: Berisi deskripsi atau ringkasan informasi mengenai lagu, termasuk genre, tema, atau narasi umum.
* **Metascore**: Merupakan skor evaluasi rata-rata yang diberikan oleh para kritikus musik terhadap lagu tersebut.
* **User Score**: Menunjukkan skor atau penilaian dari pengguna umum berdasarkan preferensi dan pengalaman mereka terhadap lagu.

Pemahaman terhadap struktur dan isi dari dataset ini menjadi langkah awal yang krusial dalam proses analisis, pemodelan, serta pengembangan sistem rekomendasi lagu berbasis *machine learning*.

---

---
### Proses analisis data
## 🔍 Eksplorasi Awal dan Pembersihan Data

Untuk memahami kondisi awal dataset, beberapa fungsi eksploratif pada *Pandas* digunakan:

* `df.info()` memberikan gambaran umum tentang struktur dataset, termasuk jumlah baris dan kolom, tipe data tiap kolom, serta jumlah nilai non-null.
* `df.isnull().sum()` digunakan untuk mengidentifikasi jumlah nilai kosong (*missing values*) pada setiap kolom.
* `df.duplicated().sum()` digunakan untuk memeriksa jumlah duplikasi baris secara keseluruhan dalam dataset.
* `df.duplicated(subset=['Name of the Song']).sum()` mengecek jumlah lagu yang memiliki judul yang sama, untuk mengetahui potensi duplikasi berdasarkan nama lagu.

### 📌 Hasil Pengamatan

Berdasarkan eksplorasi awal, diperoleh hasil sebagai berikut:

* **Jumlah total entri** dalam dataset adalah **198.126 baris**.
* **Missing values** ditemukan pada beberapa kolom:

  * `Description`: terdapat **4.369** data kosong.
  * `Metascore`: terdapat **24.385** data kosong.
  * `User Score`: terdapat **49.281** data kosong.
* **Jumlah baris yang duplikat secara keseluruhan**: **0** baris (tidak ditemukan duplikasi identik pada seluruh baris).
* **Jumlah lagu dengan nama judul yang duplikat**: sebanyak **194.214 lagu**, menunjukkan bahwa banyak lagu memiliki judul yang sama, kemungkinan besar berasal dari artis atau versi yang berbeda.

Temuan ini menjadi landasan penting untuk proses *data cleaning* dan validasi data sebelum melanjutkan ke tahap pemodelan.

---
### Visualisasi Data
Beberapa teknik visualisasi digunakan untuk memahami distribusi data antara lain:
- **Histogram**: Digunakan untuk melihat distribusi frekuensi data dalam tiap variabel.
![Histogram](histogram.png)


## 🧹 Persiapan Data untuk *Content-Based Filtering* dan *Collaborative Filtering*

---

Sebelum masuk ke tahap pemodelan, data perlu dipersiapkan dan dibersihkan agar sesuai dengan kebutuhan algoritma *machine learning*, khususnya untuk sistem rekomendasi berbasis **Content-Based Filtering** dan **Collaborative Filtering**.

### 🔧 Langkah-langkah Data Preparation:

1. **Menghapus Data yang Tidak Lengkap (Missing Values)**
   Data yang memiliki nilai kosong dihapus menggunakan fungsi `dropna()`. Hal ini penting untuk menjaga kualitas data dalam proses pelatihan model.

2. **Menghapus Duplikasi Berdasarkan Judul Lagu**
   Untuk menghindari redudansi pada hasil rekomendasi, data duplikat berdasarkan kolom **Name of the Song** dihapus menggunakan `drop_duplicates()`.

```python
# Menghapus missing values dan duplikasi berdasarkan judul lagu
df_songs.dropna(inplace=True)
df_songs.drop_duplicates(subset=['Name of the Song'], inplace=True)

# Menampilkan jumlah data setelah dibersihkan
print(f"Jumlah data setelah dibersihkan: {len(df_songs)} baris")
```

3. **Pembersihan Kolom Artist**
   Ditemukan bahwa pada kolom `Artist`, nama artis sering diawali dengan kata **"by"** (misalnya: `"by Taylor Swift"`). Untuk menjaga konsistensi dalam analisis teks dan pencocokan string, kata “by” dihapus menggunakan *regular expression*.

```python
# Salin data ke dataframe baru untuk proses encoding/pembersihan
df_songs_encode = df_songs.copy()

# Bersihkan kolom 'Artist' dengan menghapus kata 'by' di awal jika ada
df_songs_encode['Artist'] = df_songs_encode['Artist'].str.replace(r'^by\s+', '', regex=True).str.strip()

# Tampilkan hasil dataframe yang telah diproses
display(df_songs_encode.head())
```
* Setelah proses pembersihan, diperoleh **2.537 entri data lagu** dengan **7 atribut** yang siap digunakan untuk pemodelan.
* Proses pembersihan kolom `Artist` bertujuan menghindari kesalahan dalam analisis atau pencocokan nama artis.
* Dengan data yang bersih dan terstruktur, proses *Content-Based Filtering* dan *Collaborative Filtering* dapat dilakukan secara optimal dan akurat.
---
### 1. Data Preparation Content Base Filtering
Data preparation untuk Content Base Filtering didasari pada :
- Membuang atribut **Description, Unnamed: 0, dan Date of Release**. Tujuan dari pembuangan atribut ini untuk mempersiapkan atribut yang akan digunakan pada proses CBF.
- TF-IDF digunakan untuk mengubah judul lagu menjadi representasi numerik.
```python
# Salin ulang data hasil pembersihan nama artis
df_songs = df_songs_encode.copy()

# Kolom yang ingin dihapus
drop_columns = ['Description', 'Unnamed: 0', 'Date of Release']

# Hapus kolom jika memang ada dalam DataFrame
df_songs.drop(columns=[col for col in drop_columns if col in df_songs.columns], inplace=True)

# Tampilkan hasil
display(df_songs)
```
- **`TfidfVectorizer(stop_words='english')`**  
  Digunakan untuk menghilangkan kata-kata umum (seperti "the", "of", "and") agar hanya kata penting yang digunakan sebagai fitur.

- **`fit_transform(df_songs['Name of the Song'])`**  
  Melatih dan mentransformasi kolom judul lagu menjadi matriks TF-IDF.

- **`tfidf_matrix.shape` → (2537, 2858)**  
  Artinya ada **2.537 lagu unik** yang direpresentasikan oleh **2.858 kata unik** yang muncul pada judul lagu (setelah dibersihkan dan difilter).

- **`tfidf_matrix.todense()`**  
  Mengubah representasi sparse matrix (hemat memori) ke dense matrix agar dapat dilihat dan dianalisis secara eksplisit.
```python
# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # stop_words digunakan untuk menghilangkan kata umum yang tidak penting

# Transformasi kolom 'Name of the Song' menjadi representasi TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df_songs['Name of the Song'])

# Menampilkan ukuran matriks TF-IDF (baris = jumlah lagu, kolom = jumlah fitur unik dari judul lagu)
print(f"Ukuran TF-IDF Matrix: {tfidf_matrix.shape}")
```
### 2. Data Preparation Collaborative Filtering
Pada tahap ini, kita melakukan serangkaian proses transformasi dan normalisasi terhadap dataset lagu df_songs untuk mempersiapkannya sebelum digunakan dalam model prediktif seperti regresi atau neural network.

Langkah-langkah yang Dilakukan:
Normalisasi Nilai Rating (Metascore) 🎵

Skor metascore setiap lagu dinormalisasi ke dalam rentang 0 hingga 1 menggunakan min-max scaling.
Tujuan: agar model tidak bias terhadap skala nilai asli yang besar dan menjaga kestabilan selama pelatihan.
Penggabungan Fitur (X) 🎤🎶

Kolom Artist dan Name of the Song digunakan sebagai fitur numerik setelah label encoding sebelumnya.
Digabung menjadi satu array x yang kemudian dinormalisasi menggunakan MinMaxScaler agar semua fitur berada dalam skala yang seragam.
Pemisahan Dataset 📊

Dataset dibagi menjadi 80% data latih dan 20% data validasi.
Hal ini penting untuk mengevaluasi kinerja model secara obyektif terhadap data yang belum pernah dilihat sebelumnya
```python
# Target tetap menggunakan Metascore
min_rating = df_songs['Metascore'].min()
max_rating = df_songs['Metascore'].max()
y = df_songs['Metascore'].apply(lambda val: (val - min_rating) / (max_rating - min_rating)).values

# Tambahkan User Score sebagai fitur tambahan
x = df_songs[['Artist', 'Name of the Song', 'User Score']].values

# Normalisasi semua fitur
scaler_x = MinMaxScaler()
x_scaled = scaler_x.fit_transform(x)

# Split
split_index = int(0.8 * len(df_songs))
x_train, x_val = x_scaled[:split_index], x_scaled[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

print("Fitur (x):", x_scaled.shape)
print("Target (Metascore y):", y.shape)
```

## Modeling and Result
Proses modeling memuat proses perancangan model yang digunakan dalam rekomendasi.

### Content Base Filtering

```python
# Menghitung Cosine Similarity dari matrix TF-IDF
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Menampilkan ukuran matriks kemiripan (harus persegi: jumlah lagu x jumlah lagu)
print(f"Ukuran matriks Cosine Similarity: {cosine_sim.shape}")
```

### Solusi - Top 5 rekomendasi pada teknik Content Base Filtering
- Membuat DataFrame dari matriks cosine similarity
Cosine similarity digunakan untuk mengukur tingkat kemiripan antara satu lagu dengan lagu lainnya berdasarkan vektor TF-IDF dari judul lagu. Hasil dari perhitungan ini adalah matriks simetri berukuran n x n, di mana n adalah jumlah lagu. Setiap nilai dalam matriks menunjukkan tingkat kemiripan antara dua lagu. Untuk memudahkan pencarian dan penyajian rekomendasi, matriks ini kemudian diubah menjadi sebuah DataFrame, sehingga setiap lagu dapat dihubungkan langsung dengan skor kemiripannya terhadap lagu lainnya. Baris dan kolom menggunakan nama lagu untuk kemudahan interpretasi

```python
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=df_songs['Name of the Song'],
    columns=df_songs['Name of the Song']
)
```
Menampilkan ukuran dari DataFrame hasil cosine similarity
- DataFrame ini sangat penting dalam proses rekomendasi.
- Contoh: Jika pengguna menyukai lagu A, maka sistem dapat merekomendasikan lagu-lagu lain yang memiliki nilai cosine similarity tinggi terhadap lagu A.
```python
print(f" Ukuran DataFrame cosine similarity: {cosine_sim_df.shape}")
```
Menampilkan 10 sampel baris dan 10 kolom dari cosine similarity matrix
```python
cosine_sim_df.sample(n=10, axis=0).sample(n=10, axis=1)
```
Fungsi `song_recommendations()` digunakan untuk memberikan rekomendasi lagu-lagu yang memiliki kemiripan tertinggi berdasarkan **judul lagu**, menggunakan pendekatan **Content-Based Filtering (CBF)** dengan perhitungan **Cosine Similarity**.
```python
def song_recommendations(target_song, similarity_data=cosine_sim_df, items=df_songs[['Name of the Song', 'Artist']], k=5):
    if target_song not in similarity_data.columns:
        raise ValueError(f"Lagu '{target_song}' tidak ditemukan dalam data.")
    similar_scores = similarity_data[target_song].sort_values(ascending=False)[1:k+1]
    recommendations = pd.DataFrame(similar_scores).reset_index()
    recommendations.columns = ['Name of the Song', 'Similarity Score']
    recommendations = recommendations.merge(items, on='Name of the Song')
    return recommendations
```

### 🎵 Hasil Rekomendasi Menggunakan *Content-Based Filtering*

Melalui pendekatan *Content-Based Filtering*, sistem menghasilkan rekomendasi lagu berdasarkan kemiripan kata yang terkandung dalam judul lagu yang dicari oleh pengguna. Sistem ini akan menampilkan **5 lagu terdekat** yang memiliki kesamaan konten tertinggi dengan lagu referensi.

* ![Top 5 Rekomendasi](/TR1)
* ![Top 5 Rekomendasi](/TR2)
* ![Top 5 Rekomendasi](/TR3)
  
---

### Collaborative Filtering

