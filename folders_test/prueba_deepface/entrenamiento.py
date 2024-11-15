import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionResNetV2

# 1. Definir la ruta de los datos
data_dir = 'D:/Nueva carpeta/Archivos UTP/PruebaDeepface/data'  # Ruta donde están las imágenes

# 2. Crear generador de datos con aumento
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Dividir el conjunto en entrenamiento y validación
)

# 3. Cargar los datos de entrenamiento y validación
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Conjunto de entrenamiento
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Conjunto de validación
)

# 4. Configuración del modelo
# Cargar el modelo preentrenado FaceNet (InceptionResNetV2)
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

# Congelar las capas base
for layer in base_model.layers:
    layer.trainable = False

# Añadir capas personalizadas
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Número de clases

# Crear el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# 5. Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50  # Ajusta este valor según sea necesario
)

# 6. Guardar el modelo entrenado
model.save('modelos/facenet_finetuned.h5')

# 7. Evaluación del modelo
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
