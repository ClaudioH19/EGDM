# EGDM


# (1) Construir con dependencias instaladas en master y workers
docker compose build
docker compose up -d

# (2) Ejecutar en el master (usa 1 master + 2 workers automáticamente)
docker exec -it spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 /opt/spark-apps/etapa2_fase2_completo.py