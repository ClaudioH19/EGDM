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
    SAMPLE_FRACTION = float(os.environ.get("ETAPA2_SAMPLE", "0.25"))
    
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
# SECCIÓN 5: TRANSFORMACIONES DE IMÁGENES
# ==============================================================================

class ImageTransformer:
    """Transformaciones de imágenes distribuidas entre nodos."""
    
    def __init__(self, spark: SparkSession):
        """
        Inicializar transformador de imágenes.
        
        Args:
            spark: Sesión de Spark activa
        """
        self.spark = spark
        
        # Configuración de transformaciones
        self.transformations = {
            'rotation_90': {'angle': 90, 'suffix': '_rot90'},
            'rotation_180': {'angle': 180, 'suffix': '_rot180'},
            'rotation_270': {'angle': 270, 'suffix': '_rot270'},
            'brightness_increase': {'factor': 1.3, 'suffix': '_bright'},
            'brightness_decrease': {'factor': 0.7, 'suffix': '_dark'},
            'horizontal_flip': {'suffix': '_hflip'},
            'vertical_flip': {'suffix': '_vflip'},
            'zoom_in': {'factor': 1.2, 'suffix': '_zoomin'},
            'zoom_out': {'factor': 0.8, 'suffix': '_zoomout'}
        }
    
    @staticmethod
    def apply_rotation(image_path: str, angle: int, output_path: str) -> bool:
        """
        Aplicar rotación a una imagen.
        
        Args:
            image_path: Ruta de la imagen original
            angle: Ángulo de rotación en grados
            output_path: Ruta de salida
            
        Returns:
            bool: True si la transformación fue exitosa
        """
        try:
            with Image.open(image_path) as img:
                rotated = img.rotate(angle, expand=True)
                rotated.save(output_path, quality=95)
            return True
        except Exception as e:
            print(f"[TRANSFORM] Error en rotación {angle}°: {e}")
            return False
    
    @staticmethod
    def apply_brightness(image_path: str, factor: float, output_path: str) -> bool:
        """
        Ajustar brillo de una imagen.
        
        Args:
            image_path: Ruta de la imagen original
            factor: Factor de brillo (>1 aumenta, <1 disminuye)
            output_path: Ruta de salida
            
        Returns:
            bool: True si la transformación fue exitosa
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Ajustar brillo
            adjusted = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            cv2.imwrite(output_path, adjusted)
            return True
        except Exception as e:
            print(f"[TRANSFORM] Error en brillo {factor}: {e}")
            return False
    
    @staticmethod
    def apply_flip(image_path: str, flip_type: str, output_path: str) -> bool:
        """
        Aplicar inversión (flip) a una imagen.
        
        Args:
            image_path: Ruta de la imagen original
            flip_type: Tipo de inversión ('horizontal' o 'vertical')
            output_path: Ruta de salida
            
        Returns:
            bool: True si la transformación fue exitosa
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            if flip_type == 'horizontal':
                flipped = cv2.flip(img, 1)  # Flip horizontal
            elif flip_type == 'vertical':
                flipped = cv2.flip(img, 0)  # Flip vertical
            else:
                return False
            
            cv2.imwrite(output_path, flipped)
            return True
        except Exception as e:
            print(f"[TRANSFORM] Error en flip {flip_type}: {e}")
            return False
    
    @staticmethod
    def apply_zoom(image_path: str, factor: float, output_path: str) -> bool:
        """
        Aplicar zoom a una imagen.
        
        Args:
            image_path: Ruta de la imagen original
            factor: Factor de zoom (>1 acerca, <1 aleja)
            output_path: Ruta de salida
            
        Returns:
            bool: True si la transformación fue exitosa
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            h, w = img.shape[:2]
            
            if factor > 1:
                # Zoom in: recortar centro y redimensionar
                crop_h = int(h / factor)
                crop_w = int(w / factor)
                start_y = (h - crop_h) // 2
                start_x = (w - crop_w) // 2
                cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
                zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # Zoom out: redimensionar y añadir padding
                new_h = int(h * factor)
                new_w = int(w * factor)
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Crear imagen con padding
                zoomed = np.zeros((h, w, 3), dtype=np.uint8)
                start_y = (h - new_h) // 2
                start_x = (w - new_w) // 2
                zoomed[start_y:start_y + new_h, start_x:start_x + new_w] = resized
            
            cv2.imwrite(output_path, zoomed)
            return True
        except Exception as e:
            print(f"[TRANSFORM] Error en zoom {factor}: {e}")
            return False
    
    def transform_single_image(self, image_path: str) -> List[str]:
        """
        Aplicar todas las transformaciones a una sola imagen.
        
        Args:
            image_path: Ruta de la imagen a transformar
            
        Returns:
            List[str]: Lista de rutas de imágenes transformadas creadas
        """
        if not os.path.exists(image_path):
            return []
        
        base_path = os.path.splitext(image_path)[0]
        extension = os.path.splitext(image_path)[1]
        transformed_paths = []
        
        print(f"[TRANSFORM] Procesando: {os.path.basename(image_path)}")
        
        # Aplicar todas las transformaciones
        for transform_name, config in self.transformations.items():
            output_path = f"{base_path}{config['suffix']}{extension}"
            success = False
            
            if transform_name.startswith('rotation'):
                success = self.apply_rotation(image_path, config['angle'], output_path)
            elif transform_name.startswith('brightness'):
                success = self.apply_brightness(image_path, config['factor'], output_path)
            elif transform_name.endswith('flip'):
                flip_type = 'horizontal' if 'horizontal' in transform_name else 'vertical'
                success = self.apply_flip(image_path, flip_type, output_path)
            elif transform_name.startswith('zoom'):
                success = self.apply_zoom(image_path, config['factor'], output_path)
            
            if success:
                transformed_paths.append(output_path)
                print(f"[TRANSFORM]   ✓ {transform_name}: {os.path.basename(output_path)}")
            else:
                print(f"[TRANSFORM]   ✗ {transform_name}: FALLÓ")
        
        return transformed_paths
    
    def get_images_from_dataframe(self, df: DataFrame) -> Tuple[List[str], List[str]]:
        """
        Obtener imágenes del DataFrame y dividirlas entre los dos nodos disponibles.
        RESPETA EL SAMPLE_FRACTION aplicado al DataFrame.
        
        Args:
            df: DataFrame de Spark con columna 'img_path'
        
        Returns:
            Tuple[List[str], List[str]]: (imágenes_nodo1, imágenes_nodo2)
        """
        # Recopilar rutas de imágenes del DataFrame (ya filtrado por sample)
        print(f"[TRANSFORM] Obteniendo imágenes del DataFrame filtrado...")
        image_rows = df.select("img_path").collect()
        
        # Extraer rutas válidas
        original_images = [
            row["img_path"] for row in image_rows 
            if row["img_path"] is not None and os.path.exists(row["img_path"])
        ]
        
        # Dividir en dos grupos para distribución entre nodos
        total = len(original_images)
        half = total // 2
        
        node1_images = original_images[:half]
        node2_images = original_images[half:]
        
        print(f"[TRANSFORM] Distribución de trabajo:")
        print(f"[TRANSFORM]   - Nodo 1: {len(node1_images)} imágenes")
        print(f"[TRANSFORM]   - Nodo 2: {len(node2_images)} imágenes")
        print(f"[TRANSFORM]   - Total: {total} imágenes del DataFrame")
        print(f"[TRANSFORM]   - SAMPLE_FRACTION aplicado: {Config.SAMPLE_FRACTION}")
        
        return node1_images, node2_images
    
    def transform_images_distributed(self, df: DataFrame) -> None:
        """
        Ejecutar transformaciones de imágenes distribuidas entre nodos.
        RESPETA EL SAMPLE_FRACTION: solo procesa imágenes del DataFrame filtrado.
        
        Args:
            df: DataFrame de Spark con columna 'img_path' (ya filtrado por sample)
        """
        print("\n" + "="*70)
        print("PROCESAMIENTO DISTRIBUIDO DE TRANSFORMACIONES")
        print("="*70)
        
        node1_images, node2_images = self.get_images_from_dataframe(df)
        
        if not node1_images and not node2_images:
            print("[TRANSFORM] No se encontraron imágenes para transformar")
            return
        
        total_transformed = 0
        
        # Simular procesamiento en NODO 1
        if node1_images:
            print(f"\n[NODO 1] Iniciando transformaciones...")
            node1_count = 0
            for image_path in node1_images:
                transformed = self.transform_single_image(image_path)
                node1_count += len(transformed)
            
            print(f"[NODO 1] ✓ Completado: {node1_count} transformaciones generadas")
            total_transformed += node1_count
        
        # Simular procesamiento en NODO 2
        if node2_images:
            print(f"\n[NODO 2] Iniciando transformaciones...")
            node2_count = 0
            for image_path in node2_images:
                transformed = self.transform_single_image(image_path)
                node2_count += len(transformed)
            
            print(f"[NODO 2] ✓ Completado: {node2_count} transformaciones generadas")
            total_transformed += node2_count
        
        print(f"\n[TRANSFORM] ✓ TRANSFORMACIONES COMPLETADAS")
        print(f"[TRANSFORM]   - Total imágenes transformadas: {total_transformed}")
        print(f"[TRANSFORM]   - Transformaciones por imagen: {len(self.transformations)}")
        print(f"[TRANSFORM]   - Imágenes originales procesadas: {len(node1_images) + len(node2_images)}")
        
        # Verificar resultados
        self._verify_transformations()
    
    def _verify_transformations(self) -> None:
        """Verificar que las transformaciones se aplicaron correctamente."""
        print(f"\n[VERIFY] Verificando transformaciones...")
        
        # Contar imágenes transformadas
        transform_suffixes = ['_rot90', '_rot180', '_rot270', '_bright', '_dark', 
                             '_hflip', '_vflip', '_zoomin', '_zoomout']
        
        total_transformed = 0
        for suffix in transform_suffixes:
            count = 0
            for directory in [Config.PART1, Config.PART2]:
                pattern = os.path.join(directory, f"*{suffix}.*")
                count += len(glob.glob(pattern))
            
            print(f"[VERIFY]   - {suffix}: {count} imágenes")
            total_transformed += count
        
        print(f"[VERIFY] ✓ Total verificado: {total_transformed} imágenes transformadas")


# ==============================================================================
# SECCIÓN 6: ANÁLISIS EXPLORATORIO
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
# SECCIÓN 6: PREPROCESAMIENTO DE IMÁGENES (DISTRIBUIDO CON SPARK)
# ==============================================================================

class ImagePreprocessor:
    """
    Preprocesamiento distribuido de imágenes con Apache Spark.
    
    Procesa imágenes in-place (modifica las originales) usando:
    - Recorte de bordes y artefactos (reglas, viñeteo)
    - Detección y extracción del ROI del lunar
    - Atenuación del fondo para resaltar la lesión
    - Redimensionamiento a 128x128 píxeles
    
    Utiliza Spark para distribución del procesamiento entre workers.
    """
    
    @staticmethod
    def _safe_border_crop(image_rgb: np.ndarray, border_ratio: float = 0.03) -> np.ndarray:
        """Recortar bordes de la imagen para eliminar artefactos."""
        h, w = image_rgb.shape[:2]
        dy, dx = int(h * border_ratio), int(w * border_ratio)
        return image_rgb[dy:h-dy, dx:w-dx]
    
    @staticmethod
    def _auto_roi_crop(
        image_rgb: np.ndarray, 
        out_size: Tuple[int, int] = (128, 128), 
        pad_ratio: float = 0.15
    ) -> np.ndarray:
        """
        Detectar automáticamente ROI del lunar usando espacios de color Lab/HSV.
        """
        img = image_rgb
        h, w = img.shape[:2]
        
        # Convertir a espacios de color Lab y HSV
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Mapa de "oscuridad/pigmento"
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        
        score = (0.6 * (255.0 - v) + 0.3 * s + 0.1 * (a + b)).astype(np.uint8)
        
        # Umbralización con Otsu + morfología
        _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Encontrar contornos
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fallback: crop central cuadrado
        if not cnts:
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = img[y0:y0+side, x0:x0+side]
            return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
        
        # Contorno más grande (lunar)
        c = max(cnts, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        
        # Padding relativo
        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(w, x + bw + px)
        y1 = min(h, y + bh + py)
        
        crop = img[y0:y1, x0:x1]
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def _attenuate_background(image_rgb: np.ndarray, attenuation_factor: float = 0.6) -> np.ndarray:
        """Atenuar el brillo del fondo para resaltar la lesión."""
        img = image_rgb.copy()
        
        # Convertir a HSV y Lab
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
        # Crear máscara de la lesión
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)
        
        score = (0.6 * (255.0 - v) + 0.3 * s + 0.1 * (a + b)).astype(np.uint8)
        _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfología para suavizar
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Normalizar máscara
        mask_norm = mask.astype(np.float32) / 255.0
        background_mask = 1.0 - mask_norm
        
        # Aplicar atenuación
        img_float = img.astype(np.float32)
        for c in range(3):
            img_float[:, :, c] = (
                img_float[:, :, c] * mask_norm +  
                img_float[:, :, c] * background_mask * attenuation_factor
            )
        
        return np.clip(img_float, 0, 255).astype(np.uint8)
    
    @staticmethod
    def process_image_inplace(img_path: Optional[str]) -> str:
        """
        Pipeline completo de preprocesamiento que modifica la imagen original.
        Esta función será ejecutada por Spark workers de forma distribuida.
        
        Args:
            img_path: Ruta de la imagen a procesar
        
        Returns:
            Status de procesamiento: "SUCCESS", "SKIPPED", o "FAILED"
        """
        if not img_path or not os.path.exists(img_path):
            return "SKIPPED"
        
        try:
            # Leer imagen original
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return "FAILED"
            
            # Convertir BGR a RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Pipeline de procesamiento
            # Paso 1: Recortar bordes/artefactos
            rgb = ImagePreprocessor._safe_border_crop(rgb, border_ratio=0.03)
            
            # Paso 2: Auto-crop al ROI del lunar + resize a 128x128
            rgb = ImagePreprocessor._auto_roi_crop(rgb, out_size=(128, 128), pad_ratio=0.15)
            
            # Paso 3: Atenuar fondo
            rgb = ImagePreprocessor._attenuate_background(rgb, attenuation_factor=0.6)
            
            # Guardar SOBRE LA IMAGEN ORIGINAL (in-place)
            bgr_processed = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, bgr_processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return "SUCCESS"
            
        except Exception as e:
            return f"FAILED: {str(e)}"
    
    @staticmethod
    def process_dataset_distributed(df: DataFrame, spark: SparkSession) -> DataFrame:
        """
        Procesar todas las imágenes del dataset de forma distribuida con Spark.
        Modifica las imágenes originales in-place.
        
        Args:
            df: DataFrame de Spark con columna 'img_path'
            spark: Sesión de Spark activa
        
        Returns:
            DataFrame con columna adicional 'processing_status'
        """
        print(f"\n{'='*70}")
        print(f"ETAPA 5: PREPROCESAMIENTO DISTRIBUIDO DE IMÁGENES")
        print(f"{'='*70}")
        
        # Contar imágenes a procesar
        total_images = df.count()
        print(f"[PREPROCESS] Imágenes a procesar: {total_images}")
        print(f"[PREPROCESS] Tamaño objetivo: 128x128 píxeles")
        print(f"[PREPROCESS] Modo: IN-PLACE (modifica originales)")
        print(f"[PREPROCESS] Procesamiento: DISTRIBUIDO con Spark")
        
        # Crear UDF de Spark para procesamiento distribuido
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType
        
        process_udf = udf(ImagePreprocessor.process_image_inplace, StringType())
        
        # Aplicar procesamiento distribuido
        print(f"\n[PREPROCESS] Iniciando procesamiento distribuido...")
        df_processed = df.withColumn("processing_status", process_udf(col("img_path")))
        
        # Forzar ejecución y cachear resultado
        df_processed.cache()
        
        # Recopilar estadísticas
        status_counts = df_processed.groupBy("processing_status").count().collect()
        
        print(f"\n[PREPROCESS] ✓ Procesamiento completado\n")
        print(f"[PREPROCESS] Estadísticas:")
        
        success_count = 0
        for status_row in status_counts:
            status = status_row["processing_status"]
            count = status_row["count"]
            
            if status == "SUCCESS":
                success_count = count
                print(f"[PREPROCESS]   ✓ Exitosas: {count}")
            elif status == "SKIPPED":
                print(f"[PREPROCESS]   ⊘ Omitidas: {count}")
            else:
                print(f"[PREPROCESS]   ✗ Fallidas: {count}")
        
        print(f"\n[PREPROCESS] Tasa de éxito: {(success_count/total_images)*100:.2f}%")
        print(f"[PREPROCESS] Imágenes modificadas en: {Config.PART1} y {Config.PART2}\n")
        
        return df_processed


# ==============================================================================
# SECCIÓN 8: PIPELINE PRINCIPAL
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
        
        # ETAPA 5: Transformaciones de imágenes
        print("\n" + "="*70)
        print("ETAPA 5: TRANSFORMACIONES DE IMÁGENES DISTRIBUIDAS")
        print("="*70)
        
        transformer = ImageTransformer(spark)
        transformer.transform_images_distributed(df_clean)  # Pasamos el DataFrame filtrado
        
        # ETAPA 6: Preprocesamiento distribuido de imágenes
        df_final = ImagePreprocessor.process_dataset_distributed(df_clean, spark)


        # Resumen final
        print("\n" + "="*70)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"SAMPLE_FRACTION utilizado: {Config.SAMPLE_FRACTION} ({Config.SAMPLE_FRACTION*100:.1f}%)")
        print(f"Total de registros procesados: {df_final.count()}")
        print(f"Columnas: {', '.join(df_final.columns)}")
        print(f"Transformaciones de imágenes: ✓ COMPLETADAS (respetando sample)")
        print(f"Preprocesamiento: ✓ COMPLETADO (modificadas in-place)")
        print(f"Ubicación: {Config.PART1} y {Config.PART2}")
        print("="*70 + "\n")
        
    finally:
        # Cerrar sesión de Spark
        print("[SPARK] Cerrando sesión Spark...")
        spark.stop()
        print("[SPARK] ✓ Sesión cerrada\n")


if __name__ == "__main__":
    main()
