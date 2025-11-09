FROM apache/spark:4.1.0-preview3

# Usar root para instalar
USER root

# Copiar requirements.txt dentro de la imagen
COPY requirements.txt /opt/requirements.txt

ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
RUN mkdir -p /tmp/.cache/gdown && chown -R 185:0 /tmp /tmp/.cache

# Instalar pip (si hiciera falta) y dependencias
RUN apt-get update && apt-get install -y python3-pip && \
    pip install --no-cache-dir -r /opt/requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Volver al usuario de Spark (185 en la imagen oficial)
USER 185