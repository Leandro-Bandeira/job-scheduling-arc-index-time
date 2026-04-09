FROM python:3.11.13-slim

WORKDIR /app

# Atualizar repositórios e instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir a variável de ambiente para o GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copiar os arquivos de dados para o contêiner
COPY data /app/data

# Copiar o diretório src para o contêiner
COPY src /app/src

# Instalar as dependências do Python
COPY requirements.txt /app/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/main/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]