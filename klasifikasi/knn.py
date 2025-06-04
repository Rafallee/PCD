# === klasifikasi/knn.py ===
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def klasifikasi_knn(file_csv):
    df = pd.read_csv(file_csv)
    
    if 'label' not in df.columns:
        raise ValueError("Kolom 'label' tidak ditemukan dalam file CSV.")
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Visualisasi
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Aktual', marker='o')
    plt.plot(y_pred, label='Prediksi', marker='x')
    plt.title(f'Perbandingan Label Aktual vs Prediksi (Akurasi: {accuracy:.2f})')
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return accuracy
