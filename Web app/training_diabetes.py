# training_diabetes.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('diabetes_df_dataset.csv')

# Pilih kolom fitur
selected_columns = [
    'HighChol',
    'CholCheck',
    'BMI',
    'HeartDiseaseorAttack',
    'PhysHlth',
    'DiffWalk',
    'HighBP',
    'Diabetes',
    'Age_Lansia (60+)',
    'GenHlth_Fair',
    'GenHlth_Poor'
]

data = data[selected_columns]

# Pisahkan fitur dan target
X = data.drop('Diabetes', axis=1)
y = data['Diabetes']

# Split data untuk validasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Buat pipeline: scaler + logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Training pipeline
pipeline.fit(X_train, y_train)

# Simpan pipeline model
joblib.dump(pipeline, 'model_diabetes_with_Pipeline.pkl')

print("✅ Model dengan pipeline berhasil disimpan ke 'model_diabetes_with_Pipeline.pkl'")

# --- Tambahan: Validasi output kategori ---

# Contoh prediksi probabilitas untuk data testing
sample_probabilities = pipeline.predict_proba(X_test)

# Fungsi mapping kategori
def map_probability_to_category(prob):
    if prob < 0.4:
        return 'Rendah'
    elif prob < 0.7:
        return 'Sedang'
    else:
        return 'Tinggi'

# Ambil prediksi probabilitas positif (kolom 1)
positive_probs = sample_probabilities[:, 1]

# Konversi ke kategori
categories = [map_probability_to_category(prob) for prob in positive_probs]

# Tampilkan contoh hasil
print("\nContoh hasil prediksi dengan kategori:")
for prob, category in zip(positive_probs[:5], categories[:5]):
    print(f"Probabilitas: {prob:.2f} ➔ Kategori: {category}")
