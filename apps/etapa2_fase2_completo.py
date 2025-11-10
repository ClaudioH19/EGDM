"""
==============================================================================
ETAPA 2 - FASE 2: Pipeline Completo de Análisis de Datos HAM10000
==============================================================================
Pipeline de procesamiento distribuido con Apache Spark:
    1. Descarga y extracción de datos desde Google Drive
    2. Limpieza y preprocesamiento de metadatos
    3. Análisis exploratorio de datos (EDA)
    4. Generación de estadísticas y visualizaciones

Autor: Proyecto EGDM
Fecha: 2025
Requisitos: PySpark, gdown, pandas, numpy, PIL, cv2
==============================================================================
"""

import os
import zipfile
import shutil
import glob
import math
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import cv2

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import StringType, DoubleType


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN Y CONSTANTES
# ==============================================================================

class Config:
    """Configuración centralizada del proyecto."""
    
    # Configuración de entorno
    HOME = os.environ.setdefault("HOME", "/tmp")
    XDG_CACHE_HOME = os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    
    # Rutas de datos
    DATA_DIR = "/opt/spark-data"
    PART1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
    PART2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
    META = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    ZIP_PATH = os.path.join(DATA_DIR, "dataset.zip")
    
    # Google Drive
    GDRIVE_ID = "1G4ui_xWXDkhSMNpwV_3mlrgoOgIDt8TG"
    
    # Configuración de procesamiento
    SAMPLE_FRACTION = float(os.environ.get("ETAPA2_SAMPLE", "1.0"))
    
    @classmethod
    def initialize(cls):
        """Inicializar directorios necesarios."""
        os.makedirs("/tmp/.cache/gdown", exist_ok=True)
        print(f"[CONFIG] Configuración inicializada")
        print(f"[CONFIG] DATA_DIR: {cls.DATA_DIR}")
        print(f"[CONFIG] SAMPLE_FRACTION: {cls.SAMPLE_FRACTION}")


# ==============================================================================
# SECCIÓN 2: GESTIÓN DE DATOS
# ==============================================================================

class DataManager:
    """Gestión de descarga, extracción y organización de datos."""
    
    @staticmethod
    def ensure_directories() -> None:
        """Crear directorios necesarios si no existen."""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.PART1, exist_ok=True)
        os.makedirs(Config.PART2, exist_ok=True)
        print(f"[DATA] Directorios verificados/creados")
    
    @staticmethod
    def check_data_availability() -> bool:
        """
        Verificar si los datos ya están disponibles.
        
        Returns:
            bool: True si hay metadatos e imágenes disponibles
        """
        meta_exists = os.path.exists(Config.META)
        images_in_part1 = len(glob.glob(os.path.join(Config.PART1, "*.jp*g"))) > 0
        images_in_part2 = len(glob.glob(os.path.join(Config.PART2, "*.jp*g"))) > 0
        
        data_ready = meta_exists and (images_in_part1 or images_in_part2)
        
        if data_ready:
            print(f"[DATA] Datos disponibles - Metadata: ✓, Imágenes: ✓")
        
        return data_ready
    
    @staticmethod
    def download_from_gdrive() -> None:
        """Descargar dataset desde Google Drive."""
        import gdown
        
        if os.path.exists(Config.ZIP_PATH):
            print(f"[DATA] ZIP ya existe localmente: {Config.ZIP_PATH}")
            return
        
        print(f"[DATA] Descargando ZIP desde Google Drive (ID: {Config.GDRIVE_ID})...")
        gdown.download(id=Config.GDRIVE_ID, output=Config.ZIP_PATH, quiet=False, use_cookies=False)
        print(f"[DATA] Descarga completada")
    
    @staticmethod
    def extract_zip() -> None:
        """Extraer archivos del ZIP."""
        print(f"[DATA] Extrayendo ZIP: {Config.ZIP_PATH}...")
        
        with zipfile.ZipFile(Config.ZIP_PATH) as zf:
            zf.extractall(Config.DATA_DIR)
        
        print(f"[DATA] Extracción completada")
    
    @staticmethod
    def organize_metadata() -> None:
        """Mover archivo de metadatos a la ubicación correcta."""
        if os.path.exists(Config.META):
            print(f"[DATA] Metadatos ya en ubicación correcta")
            return
        
        # Buscar CSV de metadatos recursivamente
        for root, _, files in os.walk(Config.DATA_DIR):
            for filename in files:
                if filename.lower() == "ham10000_metadata.csv":
                    src = os.path.join(root, filename)
                    print(f"[DATA] Moviendo metadatos: {src} -> {Config.META}")
                    shutil.copy2(src, Config.META)
                    return
    
    @staticmethod
    def organize_images() -> None:
        """Organizar imágenes en directorios PART1 y PART2."""
        # Buscar imágenes fuera de PART1/PART2
        image_patterns = ["**/*.jpg", "**/*.JPG", "**/*.jpeg"]
        all_images = []
        
        for pattern in image_patterns:
            all_images.extend(glob.glob(os.path.join(Config.DATA_DIR, pattern), recursive=True))
        
        # Filtrar imágenes que ya están en PART1 o PART2
        images_to_organize = [
            img for img in all_images 
            if not (img.startswith(Config.PART1) or img.startswith(Config.PART2))
        ]
        
        if not images_to_organize:
            print(f"[DATA] No hay imágenes para organizar")
            return
        
        print(f"[DATA] Organizando {len(images_to_organize)} imágenes...")
        
        # Dividir en dos mitades
        half = math.ceil(len(images_to_organize) / 2)
        images_part1 = images_to_organize[:half]
        images_part2 = images_to_organize[half:]
        
        # Mover a PART1
        for img_path in images_part1:
            dst = os.path.join(Config.PART1, os.path.basename(img_path))
            shutil.move(img_path, dst)
        
        # Mover a PART2
        for img_path in images_part2:
            dst = os.path.join(Config.PART2, os.path.basename(img_path))
            shutil.move(img_path, dst)
        
        print(f"[DATA] PART1: {len(images_part1)} imágenes | PART2: {len(images_part2)} imágenes")
    
    @staticmethod
    def cleanup_zip() -> None:
        """Eliminar archivo ZIP para liberar espacio."""
        try:
            if os.path.exists(Config.ZIP_PATH):
                os.remove(Config.ZIP_PATH)
                print(f"[DATA] ZIP eliminado para liberar espacio")
        except Exception as e:
            print(f"[WARNING] No se pudo eliminar ZIP: {e}")
    
    @classmethod
    def prepare_data(cls) -> None:
        """
        Pipeline completo de preparación de datos.
        Descarga, extrae y organiza los datos si es necesario.
        """
        print("\n" + "="*70)
        print("ETAPA 1: PREPARACIÓN DE DATOS")
        print("="*70)
        
        cls.ensure_directories()
        
        if cls.check_data_availability():
            print(f"[DATA] Datos ya disponibles, omitiendo descarga")
            return
        
        cls.download_from_gdrive()
        cls.extract_zip()
        cls.organize_metadata()
        cls.organize_images()
        cls.cleanup_zip()
        
        print(f"[DATA] ✓ Preparación de datos completada\n")


# ==============================================================================
# SECCIÓN 3: UTILIDADES SPARK
# ==============================================================================

class SparkUtils:
    """Utilidades para manejo de datos con Spark."""
    
    @staticmethod
    def resolve_image_path(image_id: Optional[str]) -> Optional[str]:
        """
        Resolver la ruta completa de una imagen dado su ID.
        
        Args:
            image_id: ID de la imagen (ej: "ISIC_0024306")
        
        Returns:
            Ruta completa de la imagen o None si no se encuentra
        """
        if image_id is None:
            return None
        
        # Posibles ubicaciones y extensiones
        candidates = [
            os.path.join(Config.PART1, f"{image_id}.jpg"),
            os.path.join(Config.PART1, f"{image_id}.JPG"),
            os.path.join(Config.PART1, f"{image_id}.jpeg"),
            os.path.join(Config.PART2, f"{image_id}.jpg"),
            os.path.join(Config.PART2, f"{image_id}.JPG"),
            os.path.join(Config.PART2, f"{image_id}.jpeg"),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        return None
    
    @staticmethod
    def create_spark_session(app_name: str = "HAM10000_Pipeline") -> SparkSession:
        """
        Crear y configurar sesión de Spark.
        
        Args:
            app_name: Nombre de la aplicación Spark
        
        Returns:
            SparkSession configurada
        """
        print(f"[SPARK] Iniciando sesión Spark: {app_name}")
        
        spark = (SparkSession.builder
                 .appName(app_name)
                 .config("spark.executor.extraJavaOptions", "-Dlog4j.rootCategory=WARN,console")
                 .config("spark.driver.extraJavaOptions", "-Dlog4j.rootCategory=WARN,console")
                 .getOrCreate())
        
        spark.sparkContext.setLogLevel("WARN")
        print(f"[SPARK] ✓ Sesión Spark iniciada")
        
        return spark


# ==============================================================================
# SECCIÓN 4: PROCESAMIENTO DE DATOS
# ==============================================================================

class DataProcessor:
    """Procesamiento y limpieza de datos."""
    
    def __init__(self, spark: SparkSession):
        """
        Inicializar procesador de datos.
        
        Args:
            spark: Sesión de Spark activa
        """
        self.spark = spark
        self.resolve_path_udf = udf(SparkUtils.resolve_image_path, StringType())
    
    def load_metadata(self, sample_fraction: float = 1.0) -> DataFrame:
        """
        Cargar metadatos desde CSV.
        
        Args:
            sample_fraction: Fracción de datos a muestrear (0.0 - 1.0)
        
        Returns:
            DataFrame de Spark con metadatos
        """
        print(f"[PROCESS] Cargando metadatos desde: {Config.META}")
        
        df = (self.spark.read
              .option("header", True)
              .option("inferSchema", True)
              .csv(Config.META))
        
        # Aplicar muestreo si es necesario
        if 0 < sample_fraction < 1.0:
            df = df.sample(False, sample_fraction, seed=42)
            print(f"[PROCESS] Muestreo aplicado: {sample_fraction * 100:.1f}%")
        
        print(f"[PROCESS] ✓ Metadatos cargados: {df.count()} filas")
        return df
    
    def clean_and_enrich(self, df: DataFrame) -> DataFrame:
        """
        Limpiar datos y agregar columnas derivadas.
        
        Args:
            df: DataFrame con datos crudos
        
        Returns:
            DataFrame limpio y enriquecido
        """
        print(f"[PROCESS] Limpiando y enriqueciendo datos...")
        
        # Seleccionar columnas relevantes y castear tipos
        df_clean = (df.select("image_id", "dx", "dx_type", "sex", "age", "localization")
                    .withColumn("image_id", col("image_id").cast(StringType()))
                    .withColumn("age", col("age").cast("double"))
                    .withColumn("img_path", self.resolve_path_udf(col("image_id"))))
        
        # Contar filas antes del filtrado
        count_before = df_clean.count()
        
        # Identificar y eliminar imágenes asociadas a filas con age nulo
        null_age_images = df_clean.filter(col("age").isNull())
        null_image_paths = null_age_images.withColumn("img_path", self.resolve_path_udf(col("image_id")))
        
        # Recopilar rutas de imágenes a eliminar
        image_paths_to_delete = [
            row["img_path"] for row in null_image_paths.collect() 
            if row["img_path"] is not None
        ]
        
        # Eliminar imágenes del sistema de archivos
        for img_path in image_paths_to_delete:
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"[PROCESS] Imagen eliminada: {img_path}")
        
        # Filtrar filas con valores nulos
        df_clean = df_clean.filter(col("age").isNotNull())
        df_clean = df_clean.filter(col("img_path").isNotNull())
        
        count_after = df_clean.count()
        removed = count_before - count_after
        
        print(f"[PROCESS] ✓ Limpieza completada")
        print(f"[PROCESS]   - Filas antes: {count_before}")
        print(f"[PROCESS]   - Filas después: {count_after}")
        print(f"[PROCESS]   - Filas eliminadas: {removed}")
        
        # Verificar valores nulos restantes
        print(f"\n[PROCESS] Verificación de nulos:")
        for col_name in df_clean.columns:
            null_count = df_clean.filter(col(col_name).isNull()).count()
            print(f"[PROCESS]   - {col_name}: {null_count} nulos")
        
        return df_clean


# ==============================================================================
# SECCIÓN 5: ANÁLISIS EXPLORATORIO
# ==============================================================================

class DataAnalyzer:
    """Análisis exploratorio y generación de estadísticas."""
    
    @staticmethod
    def show_schema(df: DataFrame) -> None:
        """Mostrar esquema del DataFrame."""
        print(f"\n[ANÁLISIS] Esquema de datos:")
        df.printSchema()
    
    @staticmethod
    def show_column_stats(df: DataFrame, column: str) -> None:
        """
        Mostrar estadísticas detalladas de una columna.
        
        Args:
            df: DataFrame a analizar
            column: Nombre de la columna
        """
        print(f"\n{'='*70}")
        print(f"ESTADÍSTICAS: {column.upper()}")
        print(f"{'='*70}")
        
        total_count = df.count()
        null_count = df.filter(col(column).isNull()).count()
        
        print(f"Total de registros: {total_count}")
        print(f"Valores nulos: {null_count}")
        
        # Obtener valores distintos
        distinct_values = df.select(column).distinct().collect()
        distinct_values_list = [row[column] for row in distinct_values if row[column] is not None]
        
        print(f"Valores distintos: {distinct_values_list}")
        
        # Distribución de valores
        print(f"\nDistribución de frecuencias:")
        label_counts = df.groupBy(column).count().orderBy("count", ascending=False).collect()
        
        for row in label_counts:
            value = row[column] if row[column] is not None else "NULL"
            count = row['count']
            percentage = (count / total_count) * 100
            print(f"  {value:15s}: {count:6d} ({percentage:5.2f}%)")
    
    @staticmethod
    def generate_summary_report(df: DataFrame) -> None:
        """
        Generar reporte resumen completo del dataset.
        
        Args:
            df: DataFrame a analizar
        """
        print("\n" + "="*70)
        print("REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*70)
        
        # Análisis por columna
        columns_to_analyze = ["dx", "dx_type", "sex", "localization"]
        
        for column in columns_to_analyze:
            DataAnalyzer.show_column_stats(df, column)
        
        # Estadísticas numéricas para age
        print(f"\n{'='*70}")
        print(f"ESTADÍSTICAS NUMÉRICAS: AGE")
        print(f"{'='*70}")
        df.select("age").describe().show()
        
        print(f"\n[ANÁLISIS] ✓ Reporte completo generado\n")


# ==============================================================================
# SECCIÓN 6: PIPELINE PRINCIPAL
# ==============================================================================

def main():
    """Pipeline principal de ejecución."""
    
    # Banner inicial
    print("\n" + "="*70)
    print(" "*15 + "PIPELINE DE ANÁLISIS HAM10000")
    print("="*70 + "\n")
    
    # Inicializar configuración
    Config.initialize()
    
    # ETAPA 1: Preparación de datos
    DataManager.prepare_data()
    
    # ETAPA 2: Inicialización de Spark
    print("="*70)
    print("ETAPA 2: PROCESAMIENTO CON SPARK")
    print("="*70)
    
    spark = SparkUtils.create_spark_session()
    
    try:
        # ETAPA 3: Carga y limpieza de datos
        print("\n" + "="*70)
        print("ETAPA 3: CARGA Y LIMPIEZA DE DATOS")
        print("="*70)
        
        processor = DataProcessor(spark)
        df_raw = processor.load_metadata(sample_fraction=Config.SAMPLE_FRACTION)
        df_clean = processor.clean_and_enrich(df_raw)
        
        # ETAPA 4: Análisis exploratorio
        print("\n" + "="*70)
        print("ETAPA 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("="*70)
        
        DataAnalyzer.show_schema(df_clean)
        DataAnalyzer.generate_summary_report(df_clean)
        
        # Resumen final
        print("="*70)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"Total de registros procesados: {df_clean.count()}")
        print(f"Columnas: {', '.join(df_clean.columns)}")
        print("="*70 + "\n")
        
    finally:
        # Cerrar sesión de Spark
        print("[SPARK] Cerrando sesión Spark...")
        spark.stop()
        print("[SPARK] ✓ Sesión cerrada\n")


if __name__ == "__main__":
    main()
