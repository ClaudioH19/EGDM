# Mini workflow (descarga -> extracción -> procesamiento -> análisis)

import os, zipfile, shutil, glob, math
# ========= Librerías de imagen / numéricas =========
import numpy as np
from PIL import Image
import cv2

# ========= Spark =========
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# ========= Fix para gdown en contenedores sin HOME =========
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
os.makedirs("/tmp/.cache/gdown", exist_ok=True)

# ========= Config =========
DATA_DIR = "/opt/spark-data"
PART1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
PART2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
META = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
# ID del archivo en tu Google Drive (hazlo público con “Cualquiera con el enlace”):
GDRIVE_ID = "1G4ui_xWXDkhSMNpwV_3mlrgoOgIDt8TG"


# Fracción de muestra para la demo (puedes ajustar con ETAPA2_SAMPLE=0.25)
SAMPLE_FRACTION = float(os.environ.get("ETAPA2_SAMPLE", "1"))



# =========================
# Utilidades de datos
# =========================
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PART1, exist_ok=True)
    os.makedirs(PART2, exist_ok=True)

def have_data_ready():
    meta_ok = os.path.exists(META)
    imgs_ok = (len(glob.glob(os.path.join(PART1, "*.jp*g"))) > 0 or
               len(glob.glob(os.path.join(PART2, "*.jp*g"))) > 0)
    return meta_ok and imgs_ok

def download_and_extract_zip():
    """Descarga el ZIP de Google Drive (por ID) y lo extrae en DATA_DIR.
       Si ya hay datos o el ZIP existe, evita descargas innecesarias."""
    import gdown

    ensure_dirs()

    if have_data_ready():
        print("[INFO] Datos ya presentes. Saltando descarga.")
        return

    if os.path.exists(ZIP_PATH):
        print("[INFO] Encontrado dataset.zip local. Saltando descarga.")
    else:
        print("[INFO] Descargando ZIP desde Google Drive (ID)…")
        gdown.download(id=GDRIVE_ID, output=ZIP_PATH, quiet=False, use_cookies=False)

    print("[INFO] Extrayendo ZIP…")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    # Colocar el CSV de metadatos en la raíz si quedó en subcarpetas
    if not os.path.exists(META):
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                if f.lower() == "ham10000_metadata.csv":
                    src = os.path.join(root, f)
                    print(f"[INFO] Copiando metadatos: {src} -> {META}")
                    shutil.copy2(src, META)
                    break

    # Recolectar imágenes .jpg/.jpeg donde sea que hayan quedado
    imgs = []
    for pattern in ("**/*.jpg", "**/*.JPG", "**/*.jpeg"):
        imgs.extend(glob.glob(os.path.join(DATA_DIR, pattern), recursive=True))

    # Excluir las que ya estén dentro de part_1/part_2
    imgs = [p for p in imgs if not (p.startswith(PART1) or p.startswith(PART2))]
    print(f"[INFO] Imágenes encontradas fuera de part_1/part_2: {len(imgs)}")

    # Distribuir en dos carpetas para la demo
    if imgs:
        half = math.ceil(len(imgs) / 2)
        left, right = imgs[:half], imgs[half:]

        for p in left:
            dst = os.path.join(PART1, os.path.basename(p))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(p, dst)

        for p in right:
            dst = os.path.join(PART2, os.path.basename(p))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(p, dst)

        print(f"[INFO] Movidas a part_1: {len(left)} | a part_2: {len(right)}")

    # Limpieza del zip (opcional)
    try:
        os.remove(ZIP_PATH)
    except FileNotFoundError:
        pass

    print("[OK] Descarga y extracción completadas.")


def resolve_path(image_id: str):
    """Encuentra la ruta del .jpg de un image_id en part_1 o part_2."""
    if image_id is None:
        return None
    candidates = [
        os.path.join(PART1, f"{image_id}.jpg"),
        os.path.join(PART1, f"{image_id}.JPG"),
        os.path.join(PART1, f"{image_id}.jpeg"),
        os.path.join(PART2, f"{image_id}.jpg"),
        os.path.join(PART2, f"{image_id}.JPG"),
        os.path.join(PART2, f"{image_id}.jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def MostrarColumnasCSV(df):
    with open(META, 'r') as f:
        header = f.readline().strip()
        print("Columnas en el CSV de metadatos:")
        print(header)
    #mostrar tipo de datos
    print("Tipos de datos inferidos:")  
    df.printSchema()

    #mostar estadisticas descriptivas de valores numericos
    print("Estadísticas descriptivas de columnas numéricas:")
    numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, (DoubleType,))]
    df.select(numeric_cols).describe().show()

def MostrarEstadisticas(df, column):
    #Mostrar estadística: Cantidad de imágenes, cantidad de imágenes sin etiqueta, etiquetas distintas
    total_count = df.count()
    null_count = df.filter(col(column).isNull()).count()
    distinct_labels = df.select(column).distinct().collect()
    distinct_labels_list = [row[column] for row in distinct_labels if row[column] is
    not None]
    print(f"Total de imágenes: {total_count}")
    print(f"Imágenes sin etiqueta en '{column}': {null_count}")
    print(f"Etiquetas distintas en '{column}': {distinct_labels_list}")
    # mostrar las etiquetas de los valores de la clase objetivo
    label_counts = df.groupBy(column).count().collect()
    print(f"Conteo de etiquetas en '{column}':")
    for row in label_counts:
        print(f"  {row[column]}: {row['count']}")


    # === recorte seguro de bordes/regla/viñeteo ===
def safe_border_crop(image_rgb, border_ratio=0.03):
    h, w = image_rgb.shape[:2]
    dy, dx = int(h*border_ratio), int(w*border_ratio)
    return image_rgb[dy:h-dy, dx:w-dx]

    # === auto-crop al ROI del lunar + padding + resize ===
def auto_roi_crop(image_rgb, out_size=(128, 128), pad_ratio=0.15):
    """
    Devuelve un recorte centrado en el lunar, redimensionado a out_size.
    Heurística rápida usando Lab/HSV para encontrar pigmento/contraste.
    """
    img = image_rgb
    h, w = img.shape[:2]

    # Espacios de color
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Mapa de "oscuridad/pigmento"
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    score = (0.6*(255.0 - v) + 0.3*s + 0.1*(a + b)).astype(np.uint8)

    # Umbral + morfología
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # Contorno mayor
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Fallback: central crop cuadrado
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0:y0+side, x0:x0+side]
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)

    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    # padding relativo
    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + bw + px)
    y1 = min(h, y + bh + py)

    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)



# =========================
# Main (Spark)
# =========================
if __name__ == "__main__":
    
    # 1) Asegurar datos (descarga si faltan)
    print("[INFO] Preparando datos…")
    download_and_extract_zip()

    # 2) Spark session
    spark = (SparkSession.builder
             .appName("HAM10000_Etapa2_Completo")
             .config("spark.executor.extraJavaOptions", "-Dlog4j.rootCategory=WARN,console")
             .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=WARN,console")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    
    # 3) Lectura de metadatos + muestreo
    df_raw = (spark.read
              .option("header", True)
              .option("inferSchema", True)
              .csv(META))
    frac = max(1.0, min(SAMPLE_FRACTION, 1.0))
    df_raw = df_raw.sample(False, frac, seed=42)

    # 4) Resolver rutas de imagen y extraer features
    resolve_path_udf = udf(resolve_path, StringType())

    df = (df_raw
          .select("image_id","dx","dx_type","sex","age","localization")
          .withColumn("image_id", col("image_id").cast(StringType()))
          .withColumn("age", col("age").cast("double"))
          .withColumn("img_path", resolve_path_udf(col("image_id"))))
        
     # Identificar imágenes asociadas a filas con 'age' nulo
    null_age_images = df_raw.filter(col("age").isNull())
    null_image_paths = null_age_images.withColumn("img_path", resolve_path_udf(col("image_id")))
    
    # Recopilar las rutas de las imágenes a eliminar
    image_paths_to_delete = [row["img_path"] for row in null_image_paths.collect() if row["img_path"] is not None]

    # Eliminar las imágenes del sistema de archivos
    for img_path in image_paths_to_delete:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"[INFO] Imagen eliminada: {img_path}")

    # Filtrar las filas con 'age' nulo del DataFrame
    df = df.filter(col("age").isNotNull())
    df = df.filter(col("img_path").isNotNull())

    # Depuración: verificar filas antes y después de filtrar nulos en 'age'
    print(f"[DEBUG] Filas antes de filtrar nulos en 'age': {df.count()}")
    df = df.filter(col("age").isNotNull())
    # Verificar valores nulos después de aplicar los filtros
    print("[DEBUG] Valores nulos por columna después de filtrar:")
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        print(f"{col_name}: {null_count} nulos")

    print(f"[INFO] Filas después de eliminar nulos en 'age' y rutas de imagen: {df.count()}")

    MostrarColumnasCSV(df)
    MostrarEstadisticas(df, "dx")
    MostrarEstadisticas(df, "dx_type")
    MostrarEstadisticas(df, "sex")
    MostrarEstadisticas(df, "age")
    MostrarEstadisticas(df, "localization")
    

    ##########################################
    #####  Método 1 + 3: ROI + border crop ###
    ##########################################


    # === Directorio de salida procesada (modo fast: 128x128) ===
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed_fast_128")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Si quieres organizar por clase (dx), crea subcarpetas
    classes = [r["dx"] for r in df.select("dx").distinct().collect() if r["dx"] is not None]
    for c in classes:
        os.makedirs(os.path.join(PROCESSED_DIR, c), exist_ok=True)

    # Recolectamos un subconjunto (ya muestreaste antes con SAMPLE_FRACTION)
    rows = df.select("image_id", "dx", "img_path").collect()

    processed_count = 0
    for row in rows:
        img_path = row["img_path"]
        label = row["dx"] if row["dx"] is not None else "unknown"
        if not img_path or not os.path.exists(img_path):
            continue

        try:
            # BGR->RGB
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # M3: recorta bordes/regla
            rgb = safe_border_crop(rgb, border_ratio=0.03)

            # M1: auto-crop al ROI + resize a 128x128
            rgb = auto_roi_crop(rgb, out_size=(128, 128), pad_ratio=0.15)

            # Guardar como RGB (mantén nombre original)
            fname = os.path.basename(img_path)
            out_dir = os.path.join(PROCESSED_DIR, label)
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

            processed_count += 1
        except Exception as e:
            print(f"[WARN] Falló procesamiento {img_path}: {e}")

    print(f"[OK] Procesadas (M1+M3) {processed_count} imágenes en: {PROCESSED_DIR}")



    spark.stop()
