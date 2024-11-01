import json   
import os
import matplotlib.pyplot as plt
import numpy as np

def load_report(report_path):
    print(f"Cargando informe desde {report_path}...")
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_generate_embeddings_report(report, report_dir):
    total_images = report["total_images"]
    successful_embeddings = report.get("successful_embeddings", 0)
    failed_images = report.get("failed_images", 0)
    persons_detected = report.get("persons_detected", {})
    load_time = report.get("load_time_seconds", 0)
    embedding_time = report.get("embedding_time_seconds", 0)
    
    print("\n--- Análisis de Generate Embeddings Report ---")
    print(f"Total de imágenes: {total_images}")
    print(f"Embeddings exitosos: {successful_embeddings}")
    print(f"Imágenes fallidas: {failed_images}")
    print(f"Tiempo de carga: {load_time:.2f} segundos")
    print(f"Tiempo de generación de embeddings: {embedding_time:.2f} segundos")

    # Gráfico de personas detectadas
    plt.figure(figsize=(10, 6))
    labels = list(persons_detected.keys())
    counts = list(persons_detected.values())
    
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Personas')
    plt.ylabel('Cantidad de imágenes detectadas')
    plt.title('Cantidad de imágenes detectadas por persona')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt_path = os.path.join(report_dir, 'generate_embeddings_analysis.png')
    plt.savefig(plt_path)
    print(f"Gráfico guardado en: {plt_path}")

def analyze_train_classifier_report(report, report_dir):
    print("\n--- Análisis de Train Classifier Report ---")
    
    accuracy = report.get("accuracy", 0)
    class_report = report.get("class_report", {})
    load_time = report.get("load_time_seconds", 0)
    training_time = report.get("training_time_seconds", 0)

    num_classes = len(class_report)
    print(f"Número de clases: {num_classes}")
    
    if num_classes == 0:
        print("Advertencia: No se encontraron clases en el informe del clasificador.")
    
    print(f"Precisión del clasificador: {accuracy:.2f}")
    print(f"Tiempo de carga: {load_time:.2f} segundos")
    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    for label, metrics in class_report.items():
        print(f"{label}: Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1-Score: {metrics['f1-score']:.2f}, Support: {metrics['support']}")

    labels = list(class_report.keys())
    precision = [metrics['precision'] for metrics in class_report.values() if 'precision' in metrics]
    recall = [metrics['recall'] for metrics in class_report.values() if 'recall' in metrics]
    f1_score = [metrics['f1-score'] for metrics in class_report.values() if 'f1-score' in metrics]

    x = np.arange(len(labels))  
    width = 0.25  

    fig, ax = plt.subplots(figsize=(10, 6))

    if precision:
        ax.bar(x - width, precision, width, label='Precisión', color='blue')
    if recall:
        ax.bar(x, recall, width, label='Recall', color='orange')
    if f1_score:
        ax.bar(x + width, f1_score, width, label='F1-Score', color='green')

    ax.set_xlabel('Clases')
    ax.set_ylabel('Puntuaciones')
    ax.set_title('Análisis del clasificador por clase')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt_path = os.path.join(report_dir, 'train_classifier_analysis.png')
    plt.savefig(plt_path)
    print(f"Gráfico guardado en: {plt_path}")

    # Generar informe en formato markdown
    with open(os.path.join(report_dir, 'model_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('# Análisis del Modelo\n\n')
        f.write(f'Total de clases: {num_classes}\n\n')
        f.write(f'Precisión del clasificador: {accuracy:.2f}\n\n')
        f.write(f'Tiempo de carga: {load_time:.2f} segundos\n\n')
        f.write(f'Tiempo de entrenamiento: {training_time:.2f} segundos\n\n')
        
        f.write('## Precisión por persona\n')
        for label, metrics in class_report.items():
            if label not in ["macro avg", "weighted avg"]:
                f.write(f"- **{label}**: Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1-Score: {metrics['f1-score']:.2f}\n")
        
        # Agregar las métricas de macro avg y weighted avg
        f.write('\n## Promedio Macro y Ponderado\n')
        macro_avg = class_report.get("macro avg", {})
        weighted_avg = class_report.get("weighted avg", {})
        f.write(f"**Macro Avg** - Precision: {macro_avg['precision']:.2f}, Recall: {macro_avg['recall']:.2f}, F1-Score: {macro_avg['f1-score']:.2f}, Support: {macro_avg['support']}\n")
        f.write(f"**Weighted Avg** - Precision: {weighted_avg['precision']:.2f}, Recall: {weighted_avg['recall']:.2f}, F1-Score: {weighted_avg['f1-score']:.2f}, Support: {weighted_avg['support']}\n\n")

        # Evaluación dinámica del modelo
        f.write('## Evaluación del Modelo\n\n')
        avg_precision = np.mean(precision) if precision else 0
        if avg_precision == 1.0:
            f.write('El modelo ha demostrado un rendimiento excepcional en la clasificación de las clases evaluadas.\n\n')
        elif avg_precision >= 0.9:
            f.write('El modelo ha mostrado un buen rendimiento, con una precisión promedio superior al 90%.\n\n')
        elif avg_precision >= 0.75:
            f.write('El modelo ha demostrado un rendimiento aceptable con una precisión promedio superior al 75%.\n\n')
        else:
            f.write('El modelo ha tenido un rendimiento por debajo de las expectativas, con una precisión promedio inferior al 75%.\n\n')

        # Agregar imágenes
        f.write('\n## Análisis de Gráficos\n\n')
        f.write('### Gráfico de Personas Detectadas\n')
        f.write('![Gráfico de personas detectadas](generate_embeddings_analysis.png)\n\n')
        f.write('### Análisis del Clasificador por Clase\n')
        f.write('![Análisis del clasificador](train_classifier_analysis.png)\n\n')
        
        # Leyenda de términos
        f.write('## Leyenda de Términos\n\n')
        f.write('| Término      | Descripción                                                                                 |\n')
        f.write('|--------------|--------------------------------------------------------------------------------------------|\n')
        f.write('| **Precision**    | La proporción de verdaderos positivos sobre el total de positivos predichos.                |\n')
        f.write('| **Recall**       | La proporción de verdaderos positivos sobre el total de positivos reales.                   |\n')
        f.write('| **F1-Score**     | La media armónica entre la precisión y el recall, equilibrando ambas métricas.              |\n')
        f.write('| **Support**      | El número de ocurrencias de cada clase en el conjunto de datos.                             |\n')
        f.write('| **Macro Avg**    | Promedio simple de precisión, recall y F1-Score para cada clase, sin considerar el soporte. |\n')
        f.write('| **Weighted Avg** | Promedio ponderado de precisión, recall y F1-Score, considerando el soporte de cada clase.  |\n')


if __name__ == "__main__":
    # Rutas de los informes
    report_dir = os.path.join(os.path.dirname(__file__), '../models/informe')
    
    # Cargar y analizar los informes
    generate_embeddings_report_path = os.path.join(report_dir, 'generate_embeddings_report.json')
    train_classifier_report_path = os.path.join(report_dir, 'train_classifier_report.json')
    
    generate_embeddings_report = load_report(generate_embeddings_report_path)
    train_classifier_report = load_report(train_classifier_report_path)
    
    analyze_generate_embeddings_report(generate_embeddings_report, report_dir)
    analyze_train_classifier_report(train_classifier_report, report_dir)
