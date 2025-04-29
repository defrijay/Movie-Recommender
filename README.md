# Laporan Proyek Machine Learning - Defrizal Yahdiyan Risyad
## Project Overview

Dalam era digital, industri hiburan berkembang sangat pesat. Dengan ribuan film dirilis setiap tahunnya, penonton membutuhkan bantuan untuk menemukan film yang sesuai dengan preferensi mereka. Oleh karena itu, sistem rekomendasi film menjadi penting dalam meningkatkan pengalaman pengguna.

Masalah ini perlu diselesaikan untuk mengurangi kelelahan pengguna dalam memilih, meningkatkan kepuasan pelanggan, dan mempercepat keputusan menonton. Berdasarkan data "The Movies Dataset" dari Kaggle, proyek ini bertujuan membangun sistem rekomendasi berbasis Content-Based Filtering dan Collaborative Filtering.

Referensi:

- Ricci, F., Rokach, L., Shapira, B. (2011). Introduction to Recommender Systems Handbook. Springer.

- https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset



## Business Understanding



### Problem Statements

1. Bagaimana merekomendasikan film berdasarkan kesamaan konten dengan film yang disukai pengguna?

2. Bagaimana memberikan rekomendasi personalisasi berdasarkan interaksi pengguna lain?

### Goals

1. Membangun sistem rekomendasi berbasis konten menggunakan metadata film.

2. Membangun sistem rekomendasi berbasis interaksi pengguna menggunakan metode collaborative filtering.

### Solution Approach

1. Menggunakan Content-Based Filtering dengan cosine similarity dari fitur "genres".

2. Menggunakan Collaborative Filtering dengan pendekatan matrix factorization menggunakan SVD.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "The Movies Dataset" dari Kaggle, yang berisi dua file utama: metadata film dan rating yang diberikan oleh pengguna.


### 1. Sumber Dataset
Dataset yang digunakan dalam proyek ini adalah "The Movies Dataset" yang merupakan File-file ini berisi metadata untuk semua 45.000 film yang tercantum dalam Kumpulan Data MovieLens Lengkap. Kumpulan data ini terdiri dari film-film yang dirilis pada atau sebelum Juli 2017. Poin data meliputi pemeran, kru, kata kunci plot, anggaran, pendapatan, poster, tanggal rilis, bahasa, perusahaan produksi, negara, jumlah suara TMDB, dan rata-rata suara.

Kumpulan data ini juga memiliki file yang berisi 26 juta peringkat dari 270.000 pengguna untuk semua 45.000 film. Peringkat diberikan dalam skala 1-5 dan diperoleh dari situs web resmi GroupLens.

Sumber Sekaligus Tautan [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset).

### 2. Dimensi Dataset
Dataset terdiri dari dua bagian utama:
- **Movies Dataset** memiliki 45.466 entri dan 24 kolom.
- **Ratings Dataset** memiliki 100.004 entri dan 4 kolom.

### 3. Head Dataset
- #### Movies Dataset
    Berikut adalah contoh lima baris pertama dari dataset film:

    |     | adult | belongs_to_collection                                    | budget  | genres                                                                                     | homepage                                      | imdb_id   | original_language | original_title       | overview                                                       | release_date | revenue      | runtime | spoken_languages                               | status  | tagline                           | title                | video | vote_average | vote_count |
    |-------|-------|-----------------------------------------------------------|---------|--------------------------------------------------------------------------------------------|----------------------------------------------|-----------|-------------------|----------------------|---------------------------------------------------------------|--------------|--------------|---------|-------------------------------------------------|---------|-----------------------------------|----------------------|-------|--------------|------------|
    | 0   | False | {'id': 10194, 'name': 'Toy Story Collection', ...}         | 30000000| [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, ...]                        | http://toystory.disney.com/toy-story         | tt0114709 | en                | Toy Story            | Led by Woody, Andy's toys live happily in his ...              | 1995-10-30   | 373554033.0  | 81.0    | [{'iso_639_1': 'en', 'name': 'English'}]          | Released | NaN                               | Toy Story            | False | 7.7          | 5415.0     |
    | 1  | False | NaN                                                       | 65000000| [{'id': 12, 'name': 'Adventure'}, {'id': 14, 'name': 'Fantasy'}, ...]                        | NaN                                          | tt0113497 | en                | Jumanji              | When siblings Judy and Peter discover an encha...             | 1995-12-15   | 262797249.0 | 104.0   | [{'iso_639_1': 'en', 'name': 'English'}, {'iso...] | Released | Roll the dice and unleash the excitement! | Jumanji              | False | 6.9          | 2413.0     |
    | 2 | False | {'id': 119050, 'name': 'Grumpy Old Men Collection', ...}   | 0       | [{'id': 10749, 'name': 'Romance'}, {'id': 35, 'name': 'Comedy'}, ...]                       | NaN                                          | tt0113228 | en                | Grumpier Old Men     | A family wedding reignites the ancient feud be...             | 1995-12-22   | 0.0          | 101.0   | [{'iso_639_1': 'en', 'name': 'English'}]          | Released | Still Yelling. Still Fighting. Still Ready for... | Grumpier Old Men     | False | 6.5          | 92.0       |
    | 3 | False | NaN                                                       | 16000000| [{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}, ...]                            | NaN                                          | tt0114885 | en                | Waiting to Exhale    | Cheated on, mistreated and stepped on, the wom...              | 1995-12-22   | 81452156.0   | 127.0   | [{'iso_639_1': 'en', 'name': 'English'}]          | Released | Friends are the people who let you be yourself... | Waiting to Exhale    | False | 6.1          | 34.0       |
    | 4 | False | {'id': 96871, 'name': 'Father of the Bride Collection',...| 0       | [{'id': 35, 'name': 'Comedy'}]                                                             | NaN                                          | tt0113041 | en                | Father of the Bride Part II | Just when George Banks has recovered from his ...           | 1995-02-10   | 76578911.0  | 106.0   | [{'iso_639_1': 'en', 'name': 'English'}]          | Released | Just When His World Is Back To Normal... He's ... | Father of the Bride Part II | False | 5.7          | 173.0      |


- #### Ratings Dataset
    Berikut adalah contoh lima baris pertama dari dataset rating:

    |     | userId | movieId | rating | timestamp   |
    |-----|--------|---------|--------|-------------|
    | 0   | 1      | 31      | 2.5    | 1260759144  |
    | 1   | 1      | 1029    | 3.0    | 1260759179  |
    | 2   | 1      | 1061    | 3.0    | 1260759182  |
    | 3   | 1      | 1129    | 2.0    | 1260759185  |
    | 4   | 1      | 1172    | 4.0    | 1260759205  |


### 4. Informasi Dataset
- #### Movies Dataset
    Movies dataset memiliki 24 kolom, termasuk informasi mengenai:
    | #   | Column                | Non-Null Count | Dtype   | Penjelasan                                                                 |
    |-----|-----------------------|----------------|---------|---------------------------------------------------------------------------|
    | 0   | adult                 | 45466 non-null | object  | Menunjukkan apakah film tersebut untuk dewasa (True/False).               |
    | 1   | belongs_to_collection | 4494 non-null  | object  | Nama koleksi film jika film ini bagian dari koleksi (misalnya seri).      |
    | 2   | budget                | 45466 non-null | object  | Anggaran produksi film, biasanya dalam format numerik, tapi disimpan sebagai string. |
    | 3   | genres                | 45466 non-null | object  | Daftar genre film (misalnya, Drama, Comedy, Action).                      |
    | 4   | homepage              | 7782 non-null  | object  | URL homepage resmi film (jika ada).                                       |
    | 5   | id                    | 45466 non-null | object  | ID unik untuk setiap film.                                                |
    | 6   | imdb_id               | 45449 non-null | object  | ID IMDb unik untuk film.                                                 |
    | 7   | original_language     | 45455 non-null | object  | Bahasa asli film.                                                         |
    | 8   | original_title        | 45466 non-null | object  | Judul asli film.                                                          |
    | 9   | overview              | 44512 non-null | object  | Deskripsi atau sinopsis film.                                             |
    | 10  | popularity            | 45461 non-null | object  | Rating popularitas film berdasarkan pemirsa atau algoritma.               |
    | 11  | poster_path           | 45080 non-null | object  | URL atau path ke poster film.                                             |
    | 12  | production_companies  | 45463 non-null | object  | Daftar perusahaan produksi yang terlibat dalam pembuatan film.            |
    | 13  | production_countries  | 45463 non-null | object  | Daftar negara tempat produksi film dilakukan.                             |
    | 14  | release_date          | 45379 non-null | object  | Tanggal rilis film.                                                       |
    | 15  | revenue               | 45460 non-null | float64 | Pendapatan kotor film (dalam mata uang).                                  |
    | 16  | runtime               | 45203 non-null | float64 | Durasi film dalam menit.                                                  |
    | 17  | spoken_languages      | 45460 non-null | object  | Bahasa yang digunakan dalam film.                                         |
    | 18  | status                | 45379 non-null | object  | Status film (misalnya, Released, Post Production).                        |
    | 19  | tagline               | 20412 non-null | object  | Slogan atau tagline film.                                                 |
    | 20  | title                 | 45460 non-null | object  | Judul film.                                                               |
    | 21  | video                 | 45460 non-null | object  | Menunjukkan apakah film ini memiliki video (True/False).                  |
    | 22  | vote_average          | 45460 non-null | float64 | Rata-rata nilai suara yang diterima film.                                  |
    | 23  | vote_count            | 45460 non-null | float64 | Jumlah total suara yang diterima film.                                    |


- #### Ratings Dataset
    Ratings dataset memiliki 4 kolom:
    | #   | Column     | Non-Null Count | Dtype   | Penjelasan                                                                 |
    |-----|------------|----------------|---------|---------------------------------------------------------------------------|
    | 0   | userId     | 100004 non-null| int64   | ID unik pengguna yang memberikan rating.                                   |
    | 1   | movieId    | 100004 non-null| int64   | ID unik film yang diberikan rating.                                        |
    | 2   | rating     | 100004 non-null| float64 | Rating yang diberikan oleh pengguna pada film, dengan skala 0.5 hingga 5.  |
    | 3   | timestamp  | 100004 non-null| int64   | Waktu ketika rating diberikan, dalam format timestamp Unix.                |

### 5. Nilai yang Hilang
- #### Movies Dataset
    Beberapa kolom dalam dataset film memiliki nilai yang hilang:
    | Feature                  | Nilai yang Hilang |
    |--------------------------|-------------------|
    | adult                    | 0                 |
    | belongs_to_collection    | 40972             |
    | budget                   | 0                 |
    | genres                   | 0                 |
    | homepage                 | 37684             |
    | id                       | 0                 |
    | imdb_id                  | 17                |
    | original_language        | 11                |
    | original_title           | 0                 |
    | overview                 | 954               |
    | popularity               | 5                 |
    | poster_path              | 386               |
    | production_companies     | 3                 |
    | production_countries     | 3                 |
    | release_date             | 87                |
    | revenue                  | 6                 |
    | runtime                  | 263               |
    | spoken_languages         | 6                 |
    | status                   | 87                |
    | tagline                  | 25054             |
    | title                    | 6                 |
    | video                    | 6                 |
    | vote_average             | 6                 |
    | vote_count               | 6                 |

- #### Ratings Dataset
    Ratings dataset tidak memiliki nilai yang hilang, semua kolom memiliki 100.004 entri yang lengkap.

    | Feature    | Missing Values |
    |------------|----------------|
    | userId     | 0              |
    | movieId    | 0              |
    | rating     | 0              |
    | timestamp  | 0              |

### 6. Deskripsi Statistik Dataset
- #### Movies Dataset
    Beberapa statistik deskriptif untuk dataset film adalah sebagai berikut:

    | Kolom         | Count        | Mean         | Std           | Min      | 25%      | 50%      | 75%      | Max         |
    |---------------|--------------|--------------|---------------|----------|----------|----------|----------|-------------|
    | revenue       | 45460        | 11,209,350   | 64,332,250    | 0        | 0        | 0        | 0        | 2,787,965,000 |
    | runtime       | 45203        | 94.13        | 38.41         | 0        | 85       | 95       | 107      | 1256        |
    | vote_average  | 45460        | 5.62         | 1.92          | 0        | 5        | 6        | 6.8      | 10          |
    | vote_count    | 45460        | 109.90       | 491.31        | 0        | 3        | 10       | 34       | 14,075      |

    **Dataset Movies** berisi 45.460 entri dengan statistik utama mengenai pendapatan, durasi, rata-rata nilai suara, dan jumlah suara. Rata-rata pendapatan per film sekitar 11,2 juta, meskipun deviasi standar pendapatan cukup besar, yaitu sekitar 64,3 juta, yang menunjukkan adanya variasi yang signifikan antar film. Rata-rata durasi film adalah sekitar 94 menit, dengan deviasi standar 38 menit. Rata-rata nilai suara adalah 5,62, dan rata-rata jumlah suara sekitar 110, dengan jumlah suara maksimum mencapai 14.075. Menariknya, banyak film yang memiliki pendapatan nol (menunjukkan film dengan pendapatan rendah atau tidak tercatat), dan banyak film juga memiliki jumlah suara yang mendekati nol. Durasi film maksimum yang tercatat adalah 1.256 menit, yang sangat panjang.

- #### Ratings Dataset
    Beberapa statistik deskriptif untuk dataset rating adalah sebagai berikut:

    | Kolom         | Count        | Mean         | Std           | Min      | 25%      | 50%      | 75%      | Max         |
    |---------------|--------------|--------------|---------------|----------|----------|----------|----------|-------------|
    | userId        | 100004       | 347.01       | 195.16        | 1        | 182      | 367      | 520      | 671         |
    | movieId       | 100004       | 12,548.66    | 26,369.20     | 1        | 1,028    | 2,406    | 5,418    | 163,949     |
    | rating        | 100004       | 3.54         | 1.06          | 0.5      | 3        | 4        | 4        | 5           |
    | timestamp     | 100004       | 1,129,639,000 | 191,685,800   | 789,652,000 | 965,847,800 | 1,110,422,000 | 1,296,192,000 | 1,476,641,000 |

    **Dataset Ratings** terdiri dari 100.004 entri yang mencatat rating pengguna terhadap film. Rata-rata rating yang diberikan adalah 3,54, dengan deviasi standar 1,06, menunjukkan bahwa rating tersebar cukup merata di sekitar rata-rata. Dataset ini mencakup ID pengguna yang bervariasi dari 1 hingga 671 dan ID film dari 1 hingga 163.949. Tanggal waktu (timestamp) rating menunjukkan rentang waktu yang luas, dari sekitar 789 juta hingga 1,48 miliar, yang menunjukkan dataset ini mencakup berbagai periode waktu. Median rating adalah 4, dengan sebagian besar rating berada di antara 3 dan 4. Dataset ini memberikan wawasan penting mengenai preferensi pengguna dan distribusi rating film dari waktu ke waktu.

### **Data Preparation**

#### **1. Parsing Kolom genres**

```python
movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval)
```

Kolom genres berisi string berbentuk JSON, contoh: "[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]". Agar bisa diakses dan digunakan (misalnya untuk filter genre tertentu), perlu dikonversi menjadi list Python yang berisi nama genre. Jika tetap dalam format string, akan sulit diproses lebih lanjut.

#### **2. Menghapus Nilai Kosong pada overview**

```python
movies = movies.dropna(subset=['overview'])
```

Film yang tidak memiliki deskripsi (overview) tidak bisa diproses dalam Content-Based Filtering berbasis teks. Karena Content-Based Filtering membutuhkan representasi teks (dari overview) untuk menghitung kemiripan antar film, maka semua entri yang kosong harus dihapus agar model dapat bekerja optimal.

#### **3. Mengubah Tipe Data id Menjadi String**

```python
movies['id'] = movies['id'].astype(str)
```

Kolom id pada movies_metadata ada yang berisi karakter non-numerik. Jika dibiarkan sebagai integer, akan terjadi error ketika melakukan join/matching antar dataset (movies dan ratings). Mengubah id menjadi string membuat proses pencocokan dan manipulasi data lebih aman.

#### **4. Membuat Representasi TF-IDF dari overview**

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
```

TF-IDF (Term Frequency-Inverse Document Frequency) mengubah teks menjadi vektor numerik berdasarkan frekuensi kata. Ini diperlukan supaya deskripsi film dapat dibandingkan secara matematis menggunakan similarity metrics seperti Cosine Similarity. Tanpa TF-IDF, teks tidak bisa langsung dibandingkan.

#### **5. Menyiapkan Data Rating untuk Collaborative Filtering**

```python
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
```

Library Surprise membutuhkan format dataset khusus (Dataset object) agar bisa digunakan untuk training model Collaborative Filtering (SVD). Reader mendefinisikan skala rating, sedangkan Dataset.load_from_df mengonversi DataFrame biasa menjadi dataset yang bisa diproses Surprise.


## Modeling

### **1. Content-Based Recommendation**

Menggunakan representasi teks dari film (overview) yang diubah menjadi vektor TF-IDF. Kemudian, kemiripan antar film dihitung menggunakan Cosine Similarity.Jika pengguna menyukai sebuah film, sistem akan merekomendasikan film lain yang memiliki deskripsi yang serupa.

#### **Cara Kerja Model**

- **Representasi Film**

    Setiap film dideskripsikan menggunakan teks dari kolom overview.
    Teks ini diproses menggunakan TF-IDF Vectorizer untuk menghasilkan vektor numerik. (Vektor ini mewakili pentingnya setiap kata dalam deskripsi terhadap semua film lain.)

- **Menghitung Kemiripan**

    Setelah semua deskripsi dikonversi ke vektor TF-IDF, kita menghitung cosine similarity antar vektor. Cosine similarity mengukur sudut antara dua vektor — semakin kecil sudutnya, semakin mirip filmnya.

- **Mekanisme Rekomendasi**

    Saat pengguna memilih satu film, sistem akan:

    1. Mencari indeks film tersebut.

    2. Menghitung skor kemiripan dengan semua film lain.

    3. Mengurutkan film berdasarkan skor similarity tertinggi.

    4. Mengembalikan Top-10 film yang paling mirip.

#### **Kodenya**

1. **Membuat Similarity Matrix antar Film**

    ```python
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    ```
    linear_kernel digunakan untuk menghitung cosine similarity antara semua film berdasarkan TF-IDF yang sudah dihitung. Cosine similarity mengukur seberapa mirip dua vektor dalam ruang vektor; semakin tinggi nilai cosine similarity, semakin mirip kedua film tersebut.

2. **Mapping Judul Film ke Index**

    ```python
    indices = pd.Series(movies.index, index=movies['title']) drop_duplicates()
    ```
    Di sini, kode membuat sebuah series yang memetakan judul film ke index film dalam dataset. Ini penting karena ketika kita ingin merekomendasikan film berdasarkan judul, kita harus tahu di mana film tersebut berada dalam data.

3. **Fungsi get_recommendations**

    ```python
    def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Film tidak ditemukan."]
    idx = indices[title]
    
    sim_scores = list(enumerate(cosine_sim[idx].flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # 10 film teratas, kecuali dirinya sendiri
    movie_indices = [i[0] for i in sim_scores if i[0] < len(movies)]
    
    return movies['title'].iloc[movie_indices].tolist()
    ```
    
    Fungsi ini menerima judul film dan menghasilkan 10 rekomendasi film yang mirip berdasarkan cosine similarity.

    - **Cek keberadaan film:** Jika judul film tidak ada dalam dataset, fungsi mengembalikan pesan "Film tidak ditemukan."
    - **Indeks film:** Mencari indeks film berdasarkan judul yang diberikan.
    - **Menghitung skor kesamaan:** Mengambil nilai cosine similarity dari film yang diberikan dan membandingkannya dengan film lainnya.
    - **Mengurutkan hasil:** Menyortir skor berdasarkan kesamaan terbesar dan mengambil 10 film teratas, mengabaikan film itu sendiri.
    - **Mengambil judul film:** Menggunakan indeks film yang relevan untuk mengambil daftar judul film.

### **2. Collaborative Filtering (SVD)**

Menggunakan rating historis pengguna untuk mempelajari pola preferensi pengguna. Dengan SVD (Singular Value Decomposition), kita bisa memprediksi rating user untuk film yang belum pernah ditonton, lalu rekomendasikan film dengan prediksi tertinggi.

#### **Cara Kerja Model**

- **Data Input**

    Model ini menggunakan tabel ratings (userId, movieId, rating) sebagai input. Tidak menggunakan konten film, hanya hubungan siapa memberi rating berapa pada film tertentu.

- **Matrix Factorization**

    Collaborative Filtering membangun matrix besar:
    - Baris = pengguna
    - Kolom = film
    - Nilai = rating (jika pernah memberi rating)

- **Mekanisme Rekomendasi**

    Saat pengguna memilih satu film, sistem akan:
    1. Mencari indeks film tersebut.
    2. Menghitung skor kemiripan dengan semua film lain.
    3. Mengurutkan film berdasarkan skor similarity tertinggi.
    4. Mengembalikan Top-10 film yang paling mirip.

    Karena matrix ini sangat sparse (kebanyakan kosong karena user belum menonton semua film), SVD (Singular Value Decomposition) digunakan untuk:
    - Mereduksi dimensi matrix.
    - Mencari pola laten atau preferensi tersembunyi user dan karakteristik film.

- **Mekanisme Prediksi** 
    
    Setelah dilatih, model bisa:
    1. Memperkirakan rating yang mungkin diberikan user pada film yang belum dia tonton.
    2. Memberikan rekomendasi berdasarkan film dengan estimasi rating tertinggi.

#### **Kodenya**

1. **Membuat Model SVD**

    ```python
    model_svd = SVD()
    ```
    `SVD()` adalah model yang digunakan untuk melakukan matrix factorization dengan teknik Singular Value Decomposition. Teknik ini membagi matriks besar (seperti rating pengguna-film) menjadi matriks yang lebih kecil untuk menemukan pola tersembunyi dalam data.

2. **Melakukan Cross-Validation**

    ```python
    cross_validate(model_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    ```
    `cross_validate` digunakan untuk mengevaluasi performa model SVD dengan menggunakan teknik k-fold cross-validation (dalam hal ini, 5 fold).
    - `measures`: Model dievaluasi menggunakan dua metrik, RMSE (Root Mean Squared Error) dan MAE (Mean Absolute Error), yang keduanya mengukur akurasi prediksi model.
    - `cv=5`: Menandakan bahwa data dibagi menjadi 5 bagian (fold), di mana 4 bagian digunakan untuk pelatihan dan 1 bagian digunakan untuk pengujian, secara bergantian.
    - `verbose=True`: Menampilkan detail hasil dari proses cross-validation.

**3. Kelebihan dan Kekurangan**

| Pendekatan                     | Kelebihan                                                                 | Kekurangan                                                                 |
|---------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **Content-Based Filtering**         | - Cepat, karena tidak perlu data pengguna lain.                           | - Over-specialization (hanya rekomendasikan film yang mirip-mirip saja).   |
|                                 | - Cocok untuk cold-start user baru.                                       | - Susah rekomendasikan film dengan deskripsi yang minim.                   |
| **Collaborative Filtering (SVD)**   | - Menemukan pola tersembunyi dari interaksi user.                          | - Membutuhkan data rating cukup banyak.                                    |
|                                 | - Bisa rekomendasikan film di luar genre yang biasa ditonton.            | - Tidak efektif untuk user baru tanpa rating sebelumnya (cold-start problem). |

## Evaluation

### **1. Evaluasi Content-Based Filtering**

Metrik: `Hit Rate@10`

`Hit Rate@10` mengukur seberapa sering item yang benar-benar disukai pengguna muncul dalam 10 rekomendasi teratas yang dihasilkan oleh sistem. Metrik ini dihitung dengan rumus:

$$
\text{Hit Rate@k} = \frac{\text{Jumlah pengguna yang memiliki setidaknya 1 item relevan di top-k rekomendasi}}{\text{Jumlah total pengguna}}
$$

Pada proyek ini, diperoleh hasil:

```python
Hit Rate@10: 0.0237
```

Nilai 0.0237 menunjukkan bahwa hanya sekitar **2.37% pengguna** yang menerima rekomendasi yang sesuai dalam 10 rekomendasi teratas. Nilai ini masih tergolong rendah, menandakan bahwa pendekatan content-based masih perlu ditingkatkan, misalnya melalui pemrosesan fitur yang lebih kaya, seperti analisis NLP lanjutan atau penggunaan embedding vektor konten.


### **2. Evaluasi Content-Based Filtering**

Metrik: `RMSE dan MAE`

Pada pendekatan Collaborative Filtering dengan algoritma SVD (Singular Value Decomposition), digunakan dua metrik populer:

- **RMSE (Root Mean Square Error)** mengukur akar dari rata-rata kuadrat selisih antara rating yang diprediksi dan rating aktual:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}
$$

- **MAE (Mean Absolute Error)** mengukur rata-rata kesalahan absolut antara rating yang diprediksi dan rating aktual:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{r}_i - r_i|
$$

```python
Average RMSE: 0.8957
Average MAE : 0.6899
```

Nilai **RMSE < 1** dan **MAE < 0.7** menandakan bahwa sistem mampu memprediksi rating yang cukup mendekati nilai sebenarnya. Ini menunjukkan bahwa **Collaborative Filtering** dengan SVD bekerja lebih efektif dalam memodelkan preferensi pengguna dibandingkan pendekatan content-based pada dataset ini.
