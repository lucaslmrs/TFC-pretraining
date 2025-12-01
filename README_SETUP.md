# TFC - Time-Frequency Contrastive Learning

Guia completo para configuração e execução do TFC (Time-Frequency Contrastive Learning) para pré-treinamento auto-supervisionado de séries temporais.

## 📋 Índice

1. [Requisitos do Sistema](#requisitos-do-sistema)
2. [Instalação do Ambiente](#instalação-do-ambiente)
3. [Download dos Datasets](#download-dos-datasets)
4. [Estrutura do Projeto](#estrutura-do-projeto)
5. [Configuração](#configuração)
6. [Execução do Treinamento](#execução-do-treinamento)
7. [Cenários de Transfer Learning](#cenários-de-transfer-learning)
8. [Solução de Problemas](#solução-de-problemas)

---

## 🖥️ Requisitos do Sistema

### Hardware
| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (GTX 1650) | 8 GB+ (RTX 3070+) |
| Disco | 10 GB | 20 GB |

### Software
- **Sistema Operacional**: Linux (Ubuntu 20.04+) ou WSL2 no Windows
- **Python**: 3.9 (gerenciado via Conda)
- **CUDA**: 12.1+ (para treinamento com GPU)
- **Driver NVIDIA**: 530+ (para GPU)

### Verificar GPU (WSL2)
```bash
# No WSL, verificar se a GPU está acessível
ls /usr/lib/wsl/lib/
# Deve mostrar: libcuda.so, libnvidia-ml.so.1, nvidia-smi, etc.

# Testar nvidia-smi
/usr/lib/wsl/lib/nvidia-smi
```

---

## 🔧 Instalação do Ambiente

### Passo 1: Instalar Miniconda

```bash
# Baixar Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Instalar
bash miniconda.sh -b -p $HOME/miniconda3

# Inicializar conda no shell
~/miniconda3/bin/conda init bash

# Reiniciar o terminal ou executar:
source ~/.bashrc
```

### Passo 2: Criar Ambiente Conda

```bash
cd ~/projetos/TFC-pretraining

# Criar ambiente a partir do arquivo simplificado
conda env create -f requirements_simplified.yml

# Ativar ambiente
conda activate tfc
```

### Passo 3: Instalar PyTorch com CUDA (GPU)

```bash
conda activate tfc

# Remover PyTorch CPU e instalar com CUDA
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Passo 4: Verificar Instalação

```bash
conda activate tfc

# Verificar versões
python -c "
import torch
print(f'Python: {torch.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA disponível: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

**Saída esperada:**
```
PyTorch: 2.5.1+cu121
CUDA disponível: True
GPU: NVIDIA GeForce GTX 1650
```

---

## 📥 Download dos Datasets

### Opção 1: Script Automático

```bash
cd ~/projetos/TFC-pretraining
bash download_datasets.sh
```

### Opção 2: Download Manual (se o script falhar)

```bash
cd ~/projetos/TFC-pretraining

# Baixar cada dataset
curl -L -A "Mozilla/5.0" -o SleepEEG.zip "https://figshare.com/ndownloader/articles/19930178/versions/1"
curl -L -A "Mozilla/5.0" -o Epilepsy.zip "https://figshare.com/ndownloader/articles/19930199/versions/2"
curl -L -A "Mozilla/5.0" -o FD-A.zip "https://figshare.com/ndownloader/articles/19930205/versions/1"
curl -L -A "Mozilla/5.0" -o FD-B.zip "https://figshare.com/ndownloader/articles/19930226/versions/1"
curl -L -A "Mozilla/5.0" -o HAR.zip "https://figshare.com/ndownloader/articles/19930244/versions/1"
curl -L -A "Mozilla/5.0" -o Gesture.zip "https://figshare.com/ndownloader/articles/19930247/versions/1"
curl -L -A "Mozilla/5.0" -o ECG.zip "https://figshare.com/ndownloader/articles/19930253/versions/1"
curl -L -A "Mozilla/5.0" -o EMG.zip "https://figshare.com/ndownloader/articles/19930250/versions/1"

# Extrair
unzip SleepEEG.zip -d datasets/SleepEEG/
unzip Epilepsy.zip -d datasets/Epilepsy/
unzip FD-A.zip -d datasets/FD-A/
unzip FD-B.zip -d datasets/FD-B/
unzip HAR.zip -d datasets/HAR/
unzip Gesture.zip -d datasets/Gesture/
unzip ECG.zip -d datasets/ECG/
unzip EMG.zip -d datasets/EMG/

# Criar links simbólicos (necessário para FD_A e FD_B)
cd datasets
ln -sf FD-A FD_A
ln -sf FD-B FD_B

# Limpar arquivos zip
cd ..
rm -f *.zip
```

### Verificar Datasets

```bash
ls -lh datasets/*/train.pt
```

**Saída esperada:**
```
88K     datasets/Epilepsy/train.pt
1.4M    datasets/EMG/train.pt
1.6M    datasets/Gesture/train.pt
2.4M    datasets/FD-B/train.pt
14M     datasets/HAR/train.pt
286M    datasets/SleepEEG/train.pt
501M    datasets/ECG/train.pt
553M    datasets/FD-A/train.pt
```

---

## 📁 Estrutura do Projeto

```
TFC-pretraining/
├── datasets/                      # Datasets
│   ├── SleepEEG/                  # Pré-treino (EEG)
│   ├── Epilepsy/                  # Fine-tuning (EEG)
│   ├── HAR/                       # Pré-treino (Atividade)
│   ├── Gesture/                   # Fine-tuning (Atividade)
│   ├── FD-A/ (FD_A)               # Pré-treino (Falhas)
│   ├── FD-B/ (FD_B)               # Fine-tuning (Falhas)
│   ├── ECG/                       # Pré-treino (Cardíaco)
│   └── EMG/                       # Fine-tuning (Muscular)
│
├── code/
│   ├── TFC/                       # Código principal
│   │   ├── main.py                # Ponto de entrada
│   │   ├── model.py               # Arquitetura TFC
│   │   ├── trainer.py             # Loop de treinamento
│   │   ├── dataloader.py          # Carregamento de dados
│   │   ├── augmentations.py       # Augmentações tempo/frequência
│   │   ├── loss.py                # Função de perda NTXent
│   │   └── utils.py               # Utilitários
│   │
│   ├── config_files/              # Configurações por dataset
│   │   ├── SleepEEG_Configs.py
│   │   ├── HAR_Configs.py
│   │   ├── FD_A_Configs.py
│   │   └── ECG_Configs.py
│   │
│   └── experiments_logs/          # Logs e modelos salvos
│
├── requirements_simplified.yml    # Dependências Conda
├── download_datasets.sh           # Script de download
└── README_SETUP.md                # Este arquivo
```

---

## ⚙️ Configuração

### Arquivos de Configuração

Cada dataset tem seu arquivo em `code/config_files/`. Principais parâmetros:

```python
class Config(object):
    def __init__(self):
        # Arquitetura
        self.input_channels = 1          # Canais de entrada
        self.final_out_channels = 128    # Dimensão do embedding
        
        # Treinamento
        self.num_epoch = 40              # Épocas
        self.batch_size = 128            # Batch size (ajustar para GPU)
        self.lr = 3e-4                   # Learning rate
        
        # Dados
        self.TSlength_aligned = 178      # Comprimento da série temporal
        self.num_classes = 5             # Classes do dataset fonte
        self.num_classes_target = 2      # Classes do dataset alvo
```

### Ajustar para GPU com Pouca VRAM (4GB)

Se estiver usando GTX 1650 ou similar, reduza o batch_size:

```python
# Em config_files/FD_A_Configs.py
self.batch_size = 8           # Reduzir de 64 para 8
self.target_batch_size = 8    # Reduzir de 60 para 8
```

### Modo Debug vs Completo

No arquivo `main.py`, linha 92:

```python
# Para debug (rápido, poucos dados)
subset = True

# Para treinamento completo
subset = False
```

---

## 🚀 Execução do Treinamento

### Comandos Básicos

```bash
# Ativar ambiente
conda activate tfc

# Ir para pasta do código
cd ~/projetos/TFC-pretraining/code/TFC
```

### Fase 1: Pré-treinamento

```bash
# Sintaxe
python main.py --training_mode pre_train \
               --pretrain_dataset <DATASET_FONTE> \
               --target_dataset <DATASET_ALVO> \
               --device cuda

# Exemplo: SleepEEG → Epilepsy
python main.py --training_mode pre_train \
               --pretrain_dataset SleepEEG \
               --target_dataset Epilepsy \
               --device cuda

# Exemplo: HAR → Gesture (menor, bom para GPUs com 4GB)
python main.py --training_mode pre_train \
               --pretrain_dataset HAR \
               --target_dataset Gesture \
               --device cuda
```

### Fase 2: Fine-tuning e Teste

```bash
# Sintaxe
python main.py --training_mode fine_tune_test \
               --pretrain_dataset <DATASET_FONTE> \
               --target_dataset <DATASET_ALVO> \
               --device cuda

# Exemplo
python main.py --training_mode fine_tune_test \
               --pretrain_dataset SleepEEG \
               --target_dataset Epilepsy \
               --device cuda
```

### Parâmetros da Linha de Comando

| Parâmetro | Valores | Descrição |
|-----------|---------|-----------|
| `--training_mode` | `pre_train`, `fine_tune_test` | Modo de treinamento |
| `--pretrain_dataset` | `SleepEEG`, `HAR`, `FD_A`, `ECG` | Dataset fonte |
| `--target_dataset` | `Epilepsy`, `Gesture`, `FD_B`, `EMG` | Dataset alvo |
| `--device` | `cuda`, `cpu` | Dispositivo |
| `--seed` | `42` (default) | Seed para reprodutibilidade |
| `--logs_save_dir` | `../experiments_logs` | Diretório de logs |

---

## 🔄 Cenários de Transfer Learning

O TFC suporta 4 cenários de transferência entre domínios:

| Cenário | Pré-treino | Fine-tuning | Domínio | Tamanho | GPU 4GB |
|---------|------------|-------------|---------|---------|---------|
| 1 | SleepEEG | Epilepsy | EEG Neurológico | 286M → 88K | ✅ |
| 2 | HAR | Gesture | Reconhecimento de Atividade | 14M → 1.6M | ✅ |
| 3 | FD_A | FD_B | Detecção de Falhas | 553M → 2.4M | ❌ |
| 4 | ECG | EMG | Monitoramento Físico | 501M → 1.4M | ⚠️ |

### Recomendações por GPU

- **GTX 1650 (4GB)**: Use cenários 1 ou 2
- **RTX 3060 (8GB)**: Todos os cenários funcionam
- **RTX 3080+ (10GB+)**: Use batch_size original

---

## 🔧 Solução de Problemas

### Erro: `ModuleNotFoundError: No module named 'numpy'`

**Causa**: Ambiente conda não está ativado.

**Solução**:
```bash
conda activate tfc
```

### Erro: `CUDA out of memory`

**Causa**: GPU não tem VRAM suficiente.

**Soluções**:
1. Reduzir `batch_size` no arquivo de configuração
2. Usar dataset menor (HAR → Gesture)
3. Usar CPU: `--device cpu`

```python
# Em config_files/<Dataset>_Configs.py
self.batch_size = 8           # Reduzir
self.target_batch_size = 8    # Reduzir
```

### Erro: `FileNotFoundError: No such file or directory: '../../datasets/FD_A/train.pt'`

**Causa**: Falta link simbólico para FD_A/FD_B.

**Solução**:
```bash
cd ~/projetos/TFC-pretraining/datasets
ln -sf FD-A FD_A
ln -sf FD-B FD_B
```

### Erro: `RuntimeError: Found no NVIDIA driver`

**Causa**: Driver NVIDIA não instalado ou WSL sem suporte a GPU.

**Solução (WSL2)**:
1. Instale driver NVIDIA no Windows: https://www.nvidia.com/Download/index.aspx
2. Reinicie o computador
3. Verifique: `ls /usr/lib/wsl/lib/` deve mostrar `libcuda.so`

### Erro: `TypeError: expected Tensor as element 0, but got numpy.ndarray`

**Causa**: Dataset tem formato numpy ao invés de tensor.

**Solução**: Já corrigido no `dataloader.py`. Se persistir, verifique se está usando a versão atualizada do código.

### Erro: `ValueError: Expected more than 1 value per channel when training`

**Causa**: Batch size muito pequeno para BatchNorm.

**Solução**: Aumente `batch_size` para no mínimo 2 (recomendado: 8+).

---

## 📊 Resultados Esperados

Após o fine-tuning, você verá métricas como:

```
MLP Testing: Acc=85.00 | Precision = 84.50 | Recall = 83.20 | F1 = 83.80 | AUROC= 92.10 | AUPRC=88.50
KNN Testing: Acc=82.00 | Precision = 81.20 | Recall = 80.50 | F1 = 80.80 | AUROC= 90.30 | AUPRC=86.20
```

### Modelos Salvos

Os modelos são salvos em:
```
code/experiments_logs/<Pretrain>_2_<Target>/run1/
├── pre_train_seed_42_2layertransformer/
│   └── saved_models/
│       └── ckp_last.pt          # Checkpoint do pré-treino
└── fine_tune_test_seed_42_2layertransformer/
    └── saved_models/
        └── ckp_last.pt          # Checkpoint do fine-tuning
```

---

## 📚 Referências

- **Paper**: [Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency](https://arxiv.org/abs/2206.08496)
- **Repositório Original**: https://github.com/mims-harvard/TFC-pretraining

---

## ✅ Checklist de Instalação

- [ ] Miniconda instalado
- [ ] Ambiente `tfc` criado
- [ ] PyTorch com CUDA instalado
- [ ] GPU detectada (`torch.cuda.is_available() == True`)
- [ ] Datasets baixados e extraídos
- [ ] Links simbólicos FD_A/FD_B criados
- [ ] Pré-treinamento executado com sucesso
- [ ] Fine-tuning executado com sucesso
