# Fase 2 - Mini workflow (descarga -> extracción -> procesamiento -> análisis)
# Ejecuta un flujo con una porción pequeña de datos en 1 master + 2 workers.

import os, zipfile, shutil, glob, math

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
SAMPLE_FRACTION = float(os.environ.get("ETAPA2_SAMPLE", "0.25"))

# ========= Librerías de imagen / numéricas =========
import numpy as np
from PIL import Image

# ========= Spark =========
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


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

def MostrarColumnasCSV():
    with open(META, 'r') as f:
        header = f.readline().strip()
        print("Columnas en el CSV de metadatos:")
        print(header)
    #mostrar tipo de datos
    df = spark.read.option("header", True).option("inferSchema", True).csv(META)
    print("Tipos de datos inferidos:")  
    df.printSchema()
    #mostar valores nulos por columna
    print("Valores nulos por columna:")
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        print(f"{col_name}: {null_count} nulos")
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
    frac = max(0.01, min(SAMPLE_FRACTION, 1.0))
    df_raw = df_raw.sample(False, frac, seed=42)

    # 4) Resolver rutas de imagen y extraer features
    resolve_path_udf = udf(resolve_path, StringType())

    df = (df_raw
          .select("image_id","dx","sex","age","localization")
          .withColumn("image_id", col("image_id").cast(StringType()))
          .withColumn("age", col("age").cast("double"))
          .withColumn("age_imputed", when(col("age").isNull(), 45.0).otherwise(col("age")))
          .withColumn("img_path", resolve_path_udf(col("image_id"))))
    MostrarColumnasCSV()
    MostrarEstadisticas(df, "dx")
    MostrarEstadisticas(df, "dx_type")
    MostrarEstadisticas(df, "sex")
    MostrarEstadisticas(df, "age")
    MostrarEstadisticas(df, "localization")
    
    spark.stop()
