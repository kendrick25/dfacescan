import joblib 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from datetime import datetime
import time

def train_random_forest_classifier(embeddings, labels):
    print("Entrenando clasificador Random Forest...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Contar instancias por clase
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    print("Instancias por clase:")
    for cls, count in zip(unique_classes, counts):
        print(f"Clase {label_encoder.inverse_transform([cls])[0]}: {count}")

    # Dividir en conjunto de entrenamiento y validación
    if len(unique_classes) < 2:
        raise ValueError("Se requiere al menos 2 clases para entrenar el clasificador.")

    test_size = min(0.2, 1 / min(counts))
    X_train, X_val, y_train, y_val = train_test_split(embeddings, y_encoded, test_size=test_size, random_state=42)

    print(f"Tamaño del conjunto de entrenamiento: {len(y_train)}")
    print(f"Tamaño del conjunto de validación: {len(y_val)}")

    # Entrenar Random Forest
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    print("Entrenamiento completado. Evaluando en el conjunto de validación...")
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Precisión en el conjunto de validación: {accuracy:.2f}")

    # Generar el informe de clasificación
    target_names = label_encoder.inverse_transform(unique_classes)
    class_report = classification_report(
        y_val, 
        y_pred, 
        target_names=target_names, 
        output_dict=True
    )

    return classifier, label_encoder, accuracy, class_report

if __name__ == "__main__":
    start_time = time.time()

    # Cargar embeddings y etiquetas
    embeddings_path = os.path.join(os.path.dirname(__file__), '../models/embeddings.npy')
    labels_path = os.path.join(os.path.dirname(__file__), '../models/labels.npy')

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    load_time = time.time() - start_time
    print("Embeddings y etiquetas cargados.")

    # Verificar dimensiones de embeddings y labels
    print(f"Dimensiones de embeddings: {embeddings.shape}")
    print(f"Dimensiones de labels: {labels.shape}")

    # Entrenar clasificador y guardar modelos
    start_training_time = time.time()
    classifier, label_encoder, accuracy, class_report = train_random_forest_classifier(embeddings, labels)
    training_time = time.time() - start_training_time

    # Guardar el clasificador y el codificador de etiquetas
    classifier_path = os.path.join(os.path.dirname(__file__), '../models/face_classifier.pkl')
    label_encoder_path = os.path.join(os.path.dirname(__file__), '../models/label_encoder.pkl')
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(label_encoder, label_encoder_path)
    print("Modelo de clasificador y codificador de etiquetas guardados.")

    # Crear informe
    report = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "load_time_seconds": load_time,
        "training_time_seconds": training_time,
        "model_saved_path": os.path.abspath(classifier_path),
        "label_encoder_saved_path": os.path.abspath(label_encoder_path),
        "class_report": {}
    }

    for label in label_encoder.classes_:
        if label in class_report and label not in ["macro avg", "weighted avg"]:
            report["class_report"][label] = {
                "precision": class_report[label]["precision"],
                "recall": class_report[label]["recall"],
                "f1-score": class_report[label]["f1-score"],
                "support": class_report[label]["support"]
            }

    if "macro avg" in class_report:
        report["class_report"]["macro avg"] = class_report["macro avg"]
    if "weighted avg" in class_report:
        report["class_report"]["weighted avg"] = class_report["weighted avg"]

    report_path = os.path.join(os.path.dirname(__file__), '../models/informe/train_classifier_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    print("Modelo de clasificador Random Forest y codificador de etiquetas guardados en 'models/'. Informe guardado en 'informe/train_classifier_report.json'")
