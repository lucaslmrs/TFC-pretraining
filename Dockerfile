# ==============================================================================
# TFC-pretraining Dockerfile
# Time-Frequency Contrastive Learning for Time Series
# PyTorch 2.5.1 + CUDA 12.1 (compatibilidade exata)
# ==============================================================================

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Metadados
LABEL maintainer="lucaslmrs"
LABEL description="TFC - Time-Frequency Contrastive Learning for Time Series"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python 3.10 (padrão do Ubuntu 22.04) e dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Atualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instalar PyTorch 2.5.1 com CUDA 12.1 (versão exata)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar demais dependências
RUN pip install --no-cache-dir \
    "numpy<2" \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    einops

# Diretório de trabalho
WORKDIR /app

# Copiar código fonte
COPY code/ ./code/

# Criar diretórios para volumes
RUN mkdir -p /app/datasets /app/code/experiments_logs

# Criar links simbólicos para FD_A e FD_B
RUN ln -sf /app/datasets/FD-A /app/datasets/FD_A && \
    ln -sf /app/datasets/FD-B /app/datasets/FD_B

# Diretório de execução
WORKDIR /app/code/TFC

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Entrypoint
ENTRYPOINT ["python", "main.py"]

# Argumentos padrão (HAR → Gesture, funciona em GPUs 4GB)
CMD ["--training_mode", "pre_train", \
     "--pretrain_dataset", "HAR", \
     "--target_dataset", "Gesture", \
     "--device", "cuda"]
