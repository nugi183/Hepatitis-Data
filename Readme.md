# Laporan Proyek Machine Learning
### Nama : Anugerah Fadhil Rachman
### Nim : 211351021
### Kelas : Pagi A

## Domain Proyek

Proyek ini bertujuan untuk menyediakan informasi terperinci tentang penyakit hepatitis, termasuk penyebab, gejala, diagnosis, pengobatan, dan pencegahan. Tujuan utama proyek ini adalah untuk meningkatkan pemahaman masyarakat tentang penyakit hepatitis dan cara mencegahnya. 

## Business Understanding

Pemahaman bisnis yang kuat tentang penyakit hepatitis penting dalam mengatasi tantangan kesehatan.


### Problem Statements

- Untuk mengetahui jika seseorang terkena penyakit hepatitis. 

### Goals

- Meningkatkan pemahaman masyarakat tentang penyakit hepatitis, termasuk pencegahan, diagnosis, pengobatan, dan dampaknya pada kesehatan.

### Solution statements
- Pengembangan Platform di penyakit hepatitis Berbasis Web, Solusi pertama adalah mengembangkan platform di penyakit hepatitis berbasis web yang mengintegrasikan data dari Kaggle.com untuk memberikan pengguna akses cepat dan mudah 
- Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Dataset yang saya gunakan berasal dari Kaggle.<br> 

[Hepatitis](https://www.kaggle.com/code/nezarabdilahprakasa/hepatitis-survival-prediction-80-20-accuracy-100).


### Variabel-variabel pada Hepatitis adalah sebagai berikut:
- age : Menunjukkan umur pasien.
- sex : Menunjukkan jenis kelamin.[1 = male; 0 = female]
- steroid : kelompok senyawa kimia yang memiliki berbagai fungsi dan digunakan untuk berbagai tujuan dalam dunia medis.
- antivirals : merujuk kepada obat-obatan atau agen-agen kimia.
- fatique : medis yang mengacu pada rasa lelah atau kelemahan yang ekstrem, fisik, mental, atau emosional.
- malaise : medis yang digunakan untuk menggambarkan perasaan umum.
- anorexia : sebuah gangguan makan serius yang ditandai oleh ketakutan.
- liver_big : ungkapan yang mengacu pada pembengkakan hati, yang juga dikenal sebagai hepatomegali.
- liver_firm : merujuk kepada kondisi dan karakteristik hati.
- spleen_palpable : medis yang merujuk pada kemampuan untuk meraba atau merasakan limpa (spleen)
- spiders : merujuk pada salah satu tanda fisik yang dapat muncul pada pasien.
- ascites : medis yang digunakan untuk menggambarkan penumpukan cairan dalam rongga perut (kavum peritoneum)
- varices : kondisi medis di mana pembuluh darah vena (vena) mengalami perluasan atau pelebaran yang tidak normal.
- bilirubin : zat kimia yang dihasilkan oleh pemecahan sel darah merah tua dalam tubuh.
- alk_phosphate : sejenis enzim yang terdapat dalam tubuh manusia dan hewan.
- sgot : enzim yang terdapat dalam sel-sel hati dan jantung.
- albumin : salah satu jenis protein yang terdapat dalam darah manusia.
- protime : sebuah tes darah yang digunakan untuk mengevaluasi kemampuan darah untuk membeku.
- histology : merujuk pada studi mikroskopis jaringan hati yang terkena oleh peradangan yang terjadi akibat infeksi virus hepatitis atau kondisi lainnya.
- class : Menunjukkan live or die [Boolean, True: 1, False: 0]

## Data Preparation
### Data Collection
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Hepatitis Data, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

### Data Discovery And Profiling
Untuk bagian ini, kita akan menggunakan teknik EDA. <br>
Pertama kita mengimport semua library yang dibutuhkan,

``` bash
import pandas as pd
import numpy as np
import matplotlib.pypot as plt
import seaborn as sns
```
Lalu lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable, dan melihat 5 data paling atas di datasetnya

``` bash
data=pd.read_csv('hepatitis_csv.csv')
data.head()
```

Selanjutnya kita akan periksa datasetnya

``` bash
data.info()
```

``` bash
data.describe().T
```
Karena di dalamnya terdapat satu kolom yang tidak kita masukan, maka kita akan drop satu kolom itu

``` bash
data1=data.dropna()
data1.info()
```

``` bash
data1.columns
```

``` bash
data.drop("protime", axis = 1, inplace=True)
data.shape
```

``` bash
numeric_data = data._get_numeric_data()
numeric_data.head()
```

``` bash
numeric_data.drop('antivirals', axis=1, inplace=True)
numeric_data.drop('histology', axis=1, inplace=True)
```


``` bash
numeric_data.info()
```


``` bash
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(numeric_only=True),annot=True)
```
![Image](N1.PNG)


``` bash
plt.figure(figsize=(20, 10))
sns.displot(data.age, bins=40)
```
![Image](N2.PNG)

``` bash
sns.kdeplot(data.age)
```
![Image](N3.PNG)

``` bash
plt.figure(figsize=(20, 10))
sns.displot(data.bilirubin, bins=40)
```
![Image](N4.PNG)

``` bash
sns.kdeplot(data.bilirubin)
```
![Image](N5.PNG)

``` bash
plt.figure(figsize=(20, 10))
sns.displot(data.sgot, bins=40)
```
![Image](N6.PNG)

``` bash
sns.kdeplot(data.sgot)
```
![Image](N7.PNG)

``` bash
plt.figure(figsize=(20, 10))
sns.displot(data.alk_phosphate, bins=40)
```
![Image](N8.PNG)

``` bash
sns.kdeplot(data.alk_phosphate)
```
![Image](N9.PNG)

``` bash
plt.figure(figsize=(20, 10))
data['sex'].value_counts().plot(kind="bar", color='blue', title='Gender Distribution')
```
![Image](N10.PNG)

``` bash
data['class'].value_counts()
```


``` bash
data.isnull().sum()
```


``` bash
data.head()
```


``` bash
data = data.dropna(axis=0)
```


``` bash
data.head()
```


``` bash
data.columns
```


``` bash
data.describe()
```

## Modeling
sebelumnya mari kita import library yang nanti akan digunakan

``` bash
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

``` bash
# model.fit(X_train, y_train)
```


Langkah pertama adalah memasukkan kolom-kolom fitur yang ada di datasets dan juga kolom targetnya
``` bash
features =['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'histology']

# Menggunakan map() untuk mengubah nilai kategori menjadi numerik
encoding = {'male': 1, 'female': 2}
data['sex'] = data['sex'].map(encoding)

encoding1 = {'live': 1, 'die': 2}
data['class'] = data['class'].map(encoding1)

x = data[features]
y = data['class']
x.shape, y.shape
```

Selanjutnya kita akan menentukan berapa persen dari datasets yang akan digunakan untuk memprediksi akurasi, disini kita gunakan format akurasinya 100%
``` bash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Inisialisasi model Regresi Logistik
logistic_model = LogisticRegression()

# Latih model menggunakan data latih
logistic_model.fit(X_train, y_train)

# Lakukan prediksi menggunakan data uji
predictions = logistic_model.predict(X_test)

# Hitung akurasi prediksi
accuracy = accuracy_score(y_test, predictions)
print("Akurasi prediksi: {:.2f}%".format(accuracy * 100))

Akurasi prediksi: 82.61%
```
Wow ternyata hasil akurasi prediksi nya 82.61%,Selanjutnya kita akan memprediksi pasien hidup atau mati

``` bash
# ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'histology', 'Class']
input_data = np.array([[10, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 85, 56, 4, 0]])

# Memastikan variabel steroid diubah dari True/False menjadi 1/0
input_data[0,2] =int(input_data[0, 2])

# Memastikan variabel histology diubah dari True/False menjadi 1/0
input_data[0, -1] = int(input_data[0, -1])

# Memastikan variabel antivirals diubah dari True/False menjadi 1/0
input_data[0, 3] = int(input_data[0, 3])

# Memastikan variabel bilirubin, alk_phosphate, sgot, dan albumin diubah menjadi float
input_data[0,13:17] = input_data[0, 13:17].astype(float)

# Menggunakan Regresi Logistik untuk memprediksi "hidup" (1) atau "mati" (0)
prediction = logistic_model.predict(input_data)

# Menampilkan hasil prediksi
if prediction[0] == 1:
    print('Pasien diprediksi hidup.')
else:
    print('Pasien diprediksi mati.')

Pasien diprediksi hidup.
```
Dan ternyata hasil prediksi menyatakan pasien hidup. sekarang modelnya sudah selesai, mari kita export sebagai sav agar nanti bisa kita gunakan pada project web streamlit kita.

``` bash
import pickle

filename = "model_hepatitis.sav"
pickle.dump(logistic_model,open(filename,'wb'))
```

## Evaluation


Disini saya menggunakan Logistic Regression:

Rumus Logistic Regression adalah model matematis yang digunakan untuk memodelkan hubungan antara variabel independen (prediktor) dan variabel dependen (target) dalam bentuk yang sesuai untuk analisis klasifikasi. Dalam regresi logistik, kita menggunakan fungsi logistik atau sigmoid untuk mengubah hasil dari persamaan linier menjadi probabilitas. Berikut rumus umum dari regresi logistik: <br>
$$logit(p)=ln( 
1−p
p
​
 )=β 
0
​
 +β 
1
​
 X 
1
​
 +β 
2
​
 X 
2
​
 +…+β 
n
​
 X n$$

## Deployment


![Image](N11.PNG)