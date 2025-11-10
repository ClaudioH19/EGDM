# EGDM


# (1) Construir con dependencias instaladas en master y workers
docker compose build
docker compose up -d

# (2) Ejecutar en el master (usa 1 master + 2 workers automáticamente)
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark-apps/etapa2_fase2_completo.py


## Tareas
(1) Mostrar todas las columnas que hay en el csv, su tipo de dato, si es numérico mostrar estadísticas, mostrar valores nulos del csv (si existen)
(1.1) Buscar registros nulos y eliminar su imágen también
(2) Mostrar estadística: Cantidad de imágenes, cantidad de imágenes sin etiqueta, mostrar las etiquetas de los valores de la clase objetivo
(3) Mostrar la distribución de las clases 


NAH descartado
(3.1) Cambiar etiqueta por beningo / maligno --> transformar el problema a clasificación binaria
bcc --> malignant



(4) Por cada etiqueta minoritaria, se generará la cantidad de imágenes suficientes para alcanzar a la mayoritaria [X]

(4.a) Generar el recorte y fondo tenue en las imagenes (hecho) [Hecho]
(4.b) Ver si se puede optimizar el trabajo entre nodos (por ver) 
(4.c) Armar un script para entrenar un modelo GAN para alcanzar la cantidad de imágenes necesarias [X]

(5) Aplicar transformaciones a todo el dataset, ya sea rotaciones, aumentar y bajar brillo, inversión y zoom  



