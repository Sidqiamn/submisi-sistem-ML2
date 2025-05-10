# preprocessing/automate_sidqi.py

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Buat direktori dataset/processed jika belum ada
output_dir = 'Eksperimen_SML_Sidqi/Eksperimen_SML_Sidqi/dataset/processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data mentah
data = pd.read_csv('dataset/raw/Iris_raw.csv')


# Tampilkan kolom untuk verifikasi
print("Kolom dalam dataset:", data.columns)

# 1. Menangani missing values
for column in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    if data[column].isnull().sum() > 0:
        data[column].fillna(data[column].median(), inplace=True)

# 2. Menghapus duplikat
data.drop_duplicates(inplace=True)

# 3. Menghapus outlier pada 'SepalWidthCm'
Q1 = data['SepalWidthCm'].quantile(0.25)
Q3 = data['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['SepalWidthCm'] >= lower_bound) & (data['SepalWidthCm'] <= upper_bound)]

# 4. Encoding kolom 'Species'
label_encoder = LabelEncoder()
data['Species'] = label_encoder.fit_transform(data['Species'])

# 5. Standarisasi fitur
# Hapus kolom yang tidak relevan (misalnya, 'Id') jika ada
columns_to_drop = ['Species']
if 'Id' in data.columns:
    columns_to_drop.append('Id')

X = data.drop(columns_to_drop, axis=1)
y = data['Species']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Gabungkan dan simpan data yang sudah diproses
processed_data = pd.DataFrame(X_scaled, columns=X.columns)
processed_data['Species'] = y
processed_data.to_csv('Eksperimen_SML_Sidqi/dataset/processed/Iris_processed.csv', index=False)

print("Preprocessing selesai! Data tersimpan di Eksperimen_SML_Sidqi/dataset/processed/iris_processed.csv")