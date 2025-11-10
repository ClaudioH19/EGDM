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
import argparse
from pathlib import Path
from PIL import ImageEnhance, ImageOps

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


# ======== Data augmentation utilities (applies to whole dataset) ========
def ensure_augmented_dir(out_dir):
    os.makedirs(out_dir, exist_ok=True)


def apply_rotation(pil_img, angle):
    return pil_img.rotate(angle, resample=Image.BICUBIC, expand=False)


def apply_brightness(pil_img, factor):
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)


def apply_flip(pil_img, mode="horizontal"):
    if mode == "horizontal":
        return ImageOps.mirror(pil_img)
    elif mode == "vertical":
        return ImageOps.flip(pil_img)
    else:
        return pil_img


def apply_zoom(pil_img, zoom_factor):
    # zoom_factor >1 -> zoom in (crop center then resize back)
    # zoom_factor <1 -> zoom out (shrink then pad)
    w, h = pil_img.size
    if zoom_factor == 1.0:
        return pil_img

    if zoom_factor > 1.0:
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        right = left + new_w
        bottom = top + new_h
        cropped = pil_img.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.LANCZOS)
    else:
        # zoom out: shrink image then paste it centered on background
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        background = Image.new(pil_img.mode, (w, h))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        background.paste(resized, (left, top))
        return background


def augment_image_file(src_path, out_dir, transforms, skip_existing=True):
    """Crea y guarda versiones aumentadas de una imagen.

    transforms: dict with keys 'rotations', 'brightness', 'flips', 'zooms'
    """
    src_path = Path(src_path)
    try:
        pil_img = Image.open(src_path).convert('RGB')
    except Exception as e:
        print(f"[WARN] No se pudo abrir {src_path}: {e}")
        return 0

    base_name = src_path.stem
    ext = src_path.suffix.lower() or '.jpg'
    saved = 0

    # Always save a resized/canonical copy too
    out_base = Path(out_dir) / (base_name + ext)
    if not (skip_existing and out_base.exists()):
        pil_img.save(out_base)
        saved += 1

    # Rotations
    for angle in transforms.get('rotations', []):
        img_rot = apply_rotation(pil_img, angle)
        out_path = Path(out_dir) / f"{base_name}_rot{angle}{ext}"
        if not (skip_existing and out_path.exists()):
            img_rot.save(out_path)
            saved += 1

    # Brightness
    for factor in transforms.get('brightness', []):
        img_b = apply_brightness(pil_img, factor)
        out_path = Path(out_dir) / f"{base_name}_bright{factor:.2f}{ext}"
        if not (skip_existing and out_path.exists()):
            img_b.save(out_path)
            saved += 1

    # Flips
    for mode in transforms.get('flips', []):
        img_f = apply_flip(pil_img, mode)
        out_path = Path(out_dir) / f"{base_name}_flip{mode}{ext}"
        if not (skip_existing and out_path.exists()):
            img_f.save(out_path)
            saved += 1

    # Zooms
    for z in transforms.get('zooms', []):
        img_z = apply_zoom(pil_img, z)
        out_path = Path(out_dir) / f"{base_name}_zoom{z:.2f}{ext}"
        if not (skip_existing and out_path.exists()):
            img_z.save(out_path)
            saved += 1

    return saved


def augment_dataset(src_dirs, out_dir, transforms, exts=None, recursive=True, skip_existing=True):
    """Aplica aumentos a todos los archivos de imagen en src_dirs y guarda en out_dir."""
    ensure_augmented_dir(out_dir)
    total_saved = 0
    if exts is None:
        exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']

    patterns = []
    for d in src_dirs:
        dpath = Path(d)
        if recursive:
            patterns.extend([str(dpath / '**' / f'*{e}') for e in exts])
        else:
            patterns.extend([str(dpath / f'*{e}') for e in exts])

    seen = set()
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=recursive))

    files = sorted(set(files))
    print(f"[INFO] Encontradas {len(files)} imágenes para procesar")

    for i, fpath in enumerate(files, 1):
        rel = os.path.relpath(fpath, start=src_dirs[0] if src_dirs else '.')
        out_subdir = Path(out_dir) / Path(rel).parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        saved = augment_image_file(fpath, out_subdir, transforms, skip_existing=skip_existing)
        total_saved += saved
        if i % 100 == 0:
            print(f"[INFO] Procesadas {i}/{len(files)} imágenes, guardadas hasta ahora: {total_saved}")

    print(f"[INFO] Aumento completado. Total de archivos guardados (incluye copias): {total_saved}")
    return total_saved



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN + dataset augmentation helper")
    parser.add_argument('--augment-only', action='store_true', help='Solo aplicar aumentos al dataset y salir')
    parser.add_argument('--src-dirs', nargs='+', default=[PART1, PART2], help='Directorios fuente de imágenes (espacio-separados)')
    parser.add_argument('--out-dir', default=os.path.join(DATA_DIR, 'augmented_images'), help='Directorio donde guardar imágenes aumentadas')
    parser.add_argument('--rotations', nargs='*', type=float, default=[-15.0, 15.0], help='Ángulos de rotación en grados')
    parser.add_argument('--brightness', nargs='*', type=float, default=[0.8, 1.2], help='Factores de brillo (ej: 0.8 para bajar, 1.2 para subir)')
    parser.add_argument('--flips', nargs='*', default=['horizontal'], help='Tipos de flip: horizontal, vertical')
    parser.add_argument('--zooms', nargs='*', type=float, default=[0.9, 1.1], help='Factores de zoom (ej: 0.9, 1.1)')
    parser.add_argument('--skip-existing', action='store_true', help='No sobrescribir archivos ya existentes en el directorio de salida')
    parser.add_argument('--no-gpu', action='store_true', help='Desactivar configuración automática de GPU (útil en entornos sin TF)')

    args = parser.parse_args()

    # Intentar configurar GPU (si se usa TensorFlow y no está desactivado)
    if not args.no_gpu:
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print(f"[INFO] GPU disponible: {physical_devices[0]}")
            else:
                print("[INFO] Usando CPU para entrenamiento")
        except Exception:
            # Falló la detección/configuración de TF (p.ej. TF no instalado); continuar sin GPU
            print("[WARN] No se pudo configurar GPU o TensorFlow no disponible. Continuando sin configuración GPU.")

    # Si el usuario pide solo augmentación, ejecutarla y salir
    if args.augment_only:
        transforms = {
            'rotations': args.rotations,
            'brightness': args.brightness,
            'flips': args.flips,
            'zooms': args.zooms
        }
        print(f"[INFO] Ejecutando augmentación en: {args.src_dirs}")
        print(f"[INFO] Guardando en: {args.out_dir}")
        augment_dataset(args.src_dirs, args.out_dir, transforms, skip_existing=args.skip_existing)
    else:
        # Ejecutar flujo GAN original
        main()