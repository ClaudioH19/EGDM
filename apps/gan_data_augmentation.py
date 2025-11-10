# GAN Data Augmentation Script
# Genera imágenes sintéticas para balancear el dataset HAM10000

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from PIL import Image
import glob
from collections import Counter
import matplotlib.pyplot as plt

# ========= Configuración =========
# Detectar si estamos en contenedor Docker o entorno local
if os.path.exists("/opt/spark-data"):
    # Entorno Docker/Spark
    DATA_DIR = "/opt/spark-data"
else:
    # Entorno local - usar la carpeta data del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    DATA_DIR = os.path.join(project_root, "data")

PART1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
PART2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
META = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic_images")  # Siempre dentro de DATA_DIR

# Configuración de GAN
# Modo de entrenamiento: "fast", "balanced", "quality"
TRAINING_MODE = os.environ.get("TRAINING_MODE", "balanced")

# Configuraciones según el modo
if TRAINING_MODE == "fast":
    IMG_HEIGHT = 128  # Imágenes más pequeñas para entrenamiento rápido
    IMG_WIDTH = 128
    LATENT_DIM = 64
    BATCH_SIZE = 32
    EPOCHS_PER_CLASS = 20
    TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "1000"))  # Menos imágenes para pruebas
    print("[INFO] Modo FAST activado - Entrenamiento rápido con menor calidad")
elif TRAINING_MODE == "quality":
    IMG_HEIGHT = 450  # Tamaño original para máxima calidad
    IMG_WIDTH = 600
    LATENT_DIM = 128
    BATCH_SIZE = 8
    EPOCHS_PER_CLASS = 100
    TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "6660"))
    print("[INFO] Modo QUALITY activado - Entrenamiento lento con máxima calidad")
else:  # balanced
    IMG_HEIGHT = 224  # Tamaño balanceado
    IMG_WIDTH = 224
    LATENT_DIM = 100
    BATCH_SIZE = 16
    EPOCHS_PER_CLASS = 50
    TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "3000"))
    print("[INFO] Modo BALANCED activado - Balance entre velocidad y calidad")

print(f"[INFO] Configuración: {IMG_WIDTH}x{IMG_HEIGHT}, Batch: {BATCH_SIZE}, Épocas: {EPOCHS_PER_CLASS}")
print(f"[INFO] Objetivo: {TARGET_COUNT} imágenes por clase")

# Clases identificadas del análisis
CLASSES = ['vasc', 'nv', 'mel', 'bcc', 'df', 'bkl', 'akiec']
CLASS_COUNTS = {
    'vasc': 142,
    'nv': 6660,
    'mel': 1111,
    'bcc': 514,
    'df': 115,
    'bkl': 1089,
    'akiec': 327
}

def ensure_dirs():
    """Crear directorios necesarios para imágenes sintéticas."""
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    for class_name in CLASSES:
        class_dir = os.path.join(SYNTHETIC_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

def load_metadata():
    """Cargar metadatos del CSV."""
    return pd.read_csv(META)

def find_image_path(image_id):
    """Encuentra la ruta de una imagen dado su ID."""
    candidates = [
        os.path.join(PART1, f"{image_id}.jpg"),
        os.path.join(PART1, f"{image_id}.JPG"),
        os.path.join(PART1, f"{image_id}.jpeg"),
        os.path.join(PART2, f"{image_id}.jpg"),
        os.path.join(PART2, f"{image_id}.JPG"),
        os.path.join(PART2, f"{image_id}.jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def load_class_images(df, class_name):
    """Cargar todas las imágenes de una clase específica."""
    class_df = df[df['dx'] == class_name]
    images = []
    
    print(f"[INFO] Cargando imágenes para clase '{class_name}'...")
    
    for _, row in class_df.iterrows():
        img_path = find_image_path(row['image_id'])
        if img_path and os.path.exists(img_path):
            try:
                # Cargar imagen manteniendo el tamaño original
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img.astype(np.float32) / 255.0  # Normalizar a [0, 1]
                images.append(img)
            except Exception as e:
                print(f"[WARNING] Error cargando {img_path}: {e}")
    
    print(f"[INFO] Cargadas {len(images)} imágenes para clase '{class_name}'")
    return np.array(images)

def build_generator(latent_dim):
    """Construir el generador de la GAN con arquitectura adaptativa."""
    if TRAINING_MODE == "fast":
        # Arquitectura simple para entrenamiento rápido (128x128)
        model = keras.Sequential([
            layers.Dense(8 * 8 * 256, input_dim=latent_dim),
            layers.Reshape((8, 8, 256)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 8x8 -> 16x16
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 16x16 -> 32x32
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 32x32 -> 64x64
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 64x64 -> 128x128
            layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
    elif TRAINING_MODE == "balanced":
        # Arquitectura balanceada para 224x224
        model = keras.Sequential([
            layers.Dense(14 * 14 * 256, input_dim=latent_dim),
            layers.Reshape((14, 14, 256)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 14x14 -> 28x28
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 28x28 -> 56x56
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 56x56 -> 112x112
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 112x112 -> 224x224
            layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
    else:  # quality mode
        # Arquitectura compleja para imágenes de alta resolución (600x450)
        model = keras.Sequential([
            layers.Dense(19 * 25 * 512, input_dim=latent_dim),
            layers.Reshape((19, 25, 512)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Upsampling progresivo para llegar a 450x600
            layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Ajuste final al tamaño exacto
            layers.Lambda(lambda x: tf.image.resize(x, [IMG_HEIGHT, IMG_WIDTH])),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])
    
    return model

def build_discriminator():
    """Construir el discriminador de la GAN con arquitectura adaptativa."""
    if TRAINING_MODE == "fast":
        # Discriminador simple para 128x128
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
    else:
        # Discriminador más profundo para mayor resolución
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
    
    return model

class GAN:
    """Clase GAN para entrenar y generar imágenes."""
    
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        
        # Construir discriminador
        self.discriminator = build_discriminator()
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Construir generador
        self.generator = build_generator(latent_dim)
        
        # Construir GAN combinada
        self.discriminator.trainable = False
        gan_input = keras.Input(shape=(latent_dim,))
        generated_img = self.generator(gan_input)
        gan_output = self.discriminator(generated_img)
        
        self.gan = keras.Model(gan_input, gan_output)
        self.gan.compile(
            optimizer=keras.optimizers.Adam(0.0002, 0.5),
            loss='binary_crossentropy'
        )
    
    def train(self, X_train, epochs):
        """Entrenar la GAN."""
        batch_count = len(X_train) // BATCH_SIZE
        
        for epoch in range(epochs):
            for batch_idx in range(batch_count):
                # Entrenar discriminador
                real_imgs = X_train[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
                noise = np.random.normal(0, 1, (BATCH_SIZE, self.latent_dim))
                fake_imgs = self.generator.predict(noise, verbose=0)
                
                real_labels = np.ones((BATCH_SIZE, 1))
                fake_labels = np.zeros((BATCH_SIZE, 1))
                
                d_loss_real = self.discriminator.train_on_batch(real_imgs, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Entrenar generador
                noise = np.random.normal(0, 1, (BATCH_SIZE, self.latent_dim))
                valid_labels = np.ones((BATCH_SIZE, 1))
                g_loss = self.gan.train_on_batch(noise, valid_labels)
            
            if epoch % 20 == 0:
                print(f"[INFO] Época {epoch}/{epochs} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
    
    def generate_images(self, num_images):
        """Generar imágenes sintéticas."""
        noise = np.random.normal(0, 1, (num_images, self.latent_dim))
        generated_imgs = self.generator.predict(noise, verbose=0)
        return generated_imgs

def train_gan_for_class(class_name, df):
    """Entrenar GAN para una clase específica y generar imágenes."""
    print(f"\n{'='*50}")
    print(f"[INFO] Procesando clase: {class_name}")
    
    current_count = CLASS_COUNTS[class_name]
    needed_images = TARGET_COUNT - current_count
    
    if needed_images <= 0:
        print(f"[INFO] Clase '{class_name}' ya tiene suficientes imágenes ({current_count})")
        return
    
    print(f"[INFO] Imágenes actuales: {current_count}")
    print(f"[INFO] Imágenes necesarias: {needed_images}")
    
    # Cargar imágenes de la clase
    class_images = load_class_images(df, class_name)
    
    if len(class_images) == 0:
        print(f"[ERROR] No se encontraron imágenes para la clase '{class_name}'")
        return
    
    # Crear y entrenar GAN
    print(f"[INFO] Entrenando GAN para clase '{class_name}'...")
    gan = GAN(LATENT_DIM)
    gan.train(class_images, EPOCHS_PER_CLASS)
    
    # Generar imágenes sintéticas
    print(f"[INFO] Generando {needed_images} imágenes sintéticas...")
    synthetic_images = gan.generate_images(needed_images)
    
    # Guardar imágenes sintéticas
    class_dir = os.path.join(SYNTHETIC_DIR, class_name)
    for i, img in enumerate(synthetic_images):
        # Desnormalizar imagen (de [0,1] a [0,255])
        img_scaled = (img * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_scaled)
        
        # Nombre con identificador sintético
        filename = f"synthetic_{class_name}_{i+1:05d}.jpg"
        filepath = os.path.join(class_dir, filename)
        img_pil.save(filepath)
    
    print(f"[INFO] Guardadas {needed_images} imágenes sintéticas en {class_dir}")

def main():
    """Función principal."""
    print("[INFO] Iniciando generación de imágenes sintéticas con GAN...")
    print(f"[INFO] Objetivo: {TARGET_COUNT} imágenes por clase")
    
    # Configurar directorios
    ensure_dirs()
    
    # Cargar metadatos
    print("[INFO] Cargando metadatos...")
    df = load_metadata()
    
    # Filtrar filas con age no nulo (similar al script de Spark)
    df = df.dropna(subset=['age'])
    print(f"[INFO] Dataset limpio: {len(df)} filas")
    
    # Verificar distribución actual
    class_distribution = df['dx'].value_counts()
    print("\n[INFO] Distribución actual de clases:")
    for class_name in CLASSES:
        count = class_distribution.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    # Entrenar GAN para cada clase que necesite más imágenes
    for class_name in CLASSES:
        if CLASS_COUNTS[class_name] < TARGET_COUNT:
            train_gan_for_class(class_name, df)
        else:
            print(f"[INFO] Saltando clase '{class_name}' - ya tiene suficientes imágenes")
    
    print("\n[INFO] ¡Generación de imágenes sintéticas completada!")
    print(f"[INFO] Las imágenes sintéticas están guardadas en: {SYNTHETIC_DIR}")

if __name__ == "__main__":
    # Configurar GPU si está disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"[INFO] GPU disponible: {physical_devices[0]}")
    else:
        print("[INFO] Usando CPU para entrenamiento")
    
    main()