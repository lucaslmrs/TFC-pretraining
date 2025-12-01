# ==============================================================================
# TFC-pretraining Makefile
# Comandos simplificados para build e execução
# ==============================================================================

# Configurações padrão (podem ser sobrescritas na linha de comando)
PRETRAIN_DATASET ?= HAR
TARGET_DATASET ?= Gesture
DEVICE ?= cuda
SEED ?= 42

# Nome da imagem Docker
IMAGE_NAME = tfc-pretraining
IMAGE_TAG = latest

# Cores para output
GREEN = \033[0;32m
YELLOW = \033[0;33m
NC = \033[0m

# ==============================================================================
# Build
# ==============================================================================

.PHONY: build
build: ## Construir imagem Docker
	@echo "$(GREEN)🔨 Construindo imagem Docker...$(NC)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# ==============================================================================
# Download de Datasets
# ==============================================================================

.PHONY: download-data
download-data: ## Baixar todos os datasets
	@echo "$(GREEN)📥 Baixando datasets...$(NC)"
	@mkdir -p datasets
	@echo "Baixando SleepEEG..."
	@curl -L -A "Mozilla/5.0" -o SleepEEG.zip "https://figshare.com/ndownloader/articles/19930178/versions/1"
	@echo "Baixando Epilepsy..."
	@curl -L -A "Mozilla/5.0" -o Epilepsy.zip "https://figshare.com/ndownloader/articles/19930199/versions/2"
	@echo "Baixando FD-A..."
	@curl -L -A "Mozilla/5.0" -o FD-A.zip "https://figshare.com/ndownloader/articles/19930205/versions/1"
	@echo "Baixando FD-B..."
	@curl -L -A "Mozilla/5.0" -o FD-B.zip "https://figshare.com/ndownloader/articles/19930226/versions/1"
	@echo "Baixando HAR..."
	@curl -L -A "Mozilla/5.0" -o HAR.zip "https://figshare.com/ndownloader/articles/19930244/versions/1"
	@echo "Baixando Gesture..."
	@curl -L -A "Mozilla/5.0" -o Gesture.zip "https://figshare.com/ndownloader/articles/19930247/versions/1"
	@echo "Baixando ECG..."
	@curl -L -A "Mozilla/5.0" -o ECG.zip "https://figshare.com/ndownloader/articles/19930253/versions/1"
	@echo "Baixando EMG..."
	@curl -L -A "Mozilla/5.0" -o EMG.zip "https://figshare.com/ndownloader/articles/19930250/versions/1"
	@echo "$(GREEN)📦 Extraindo arquivos...$(NC)"
	@unzip -o SleepEEG.zip -d datasets/SleepEEG/
	@unzip -o Epilepsy.zip -d datasets/Epilepsy/
	@unzip -o FD-A.zip -d datasets/FD-A/
	@unzip -o FD-B.zip -d datasets/FD-B/
	@unzip -o HAR.zip -d datasets/HAR/
	@unzip -o Gesture.zip -d datasets/Gesture/
	@unzip -o ECG.zip -d datasets/ECG/
	@unzip -o EMG.zip -d datasets/EMG/
	@echo "$(GREEN)🔗 Criando links simbólicos...$(NC)"
	@cd datasets && ln -sf FD-A FD_A && ln -sf FD-B FD_B
	@echo "$(GREEN)🧹 Limpando arquivos zip...$(NC)"
	@rm -f *.zip
	@echo "$(GREEN)✅ Datasets baixados com sucesso!$(NC)"

# ==============================================================================
# Treinamento com GPU
# ==============================================================================

.PHONY: pretrain
pretrain: ## Pré-treinamento com GPU
	@echo "$(GREEN)🚀 Iniciando pré-treinamento (GPU)...$(NC)"
	@echo "   Dataset fonte: $(PRETRAIN_DATASET)"
	@echo "   Dataset alvo: $(TARGET_DATASET)"
	docker compose --profile gpu run --rm tfc-gpu \
		--training_mode pre_train \
		--pretrain_dataset $(PRETRAIN_DATASET) \
		--target_dataset $(TARGET_DATASET) \
		--device cuda \
		--seed $(SEED)

.PHONY: finetune
finetune: ## Fine-tuning e teste com GPU
	@echo "$(GREEN)🎯 Iniciando fine-tuning (GPU)...$(NC)"
	@echo "   Dataset fonte: $(PRETRAIN_DATASET)"
	@echo "   Dataset alvo: $(TARGET_DATASET)"
	docker compose --profile gpu run --rm tfc-gpu \
		--training_mode fine_tune_test \
		--pretrain_dataset $(PRETRAIN_DATASET) \
		--target_dataset $(TARGET_DATASET) \
		--device cuda \
		--seed $(SEED)

.PHONY: train-full
train-full: pretrain finetune ## Pipeline completo: pré-treino + fine-tuning
	@echo "$(GREEN)✅ Pipeline completo finalizado!$(NC)"

# ==============================================================================
# Treinamento com CPU
# ==============================================================================

.PHONY: pretrain-cpu
pretrain-cpu: ## Pré-treinamento com CPU
	@echo "$(YELLOW)🐢 Iniciando pré-treinamento (CPU - mais lento)...$(NC)"
	docker compose --profile cpu run --rm tfc-cpu \
		--training_mode pre_train \
		--pretrain_dataset $(PRETRAIN_DATASET) \
		--target_dataset $(TARGET_DATASET) \
		--device cpu \
		--seed $(SEED)

.PHONY: finetune-cpu
finetune-cpu: ## Fine-tuning e teste com CPU
	@echo "$(YELLOW)🐢 Iniciando fine-tuning (CPU - mais lento)...$(NC)"
	docker compose --profile cpu run --rm tfc-cpu \
		--training_mode fine_tune_test \
		--pretrain_dataset $(PRETRAIN_DATASET) \
		--target_dataset $(TARGET_DATASET) \
		--device cpu \
		--seed $(SEED)

# ==============================================================================
# Cenários Pré-definidos
# ==============================================================================

.PHONY: scenario-eeg
scenario-eeg: ## Cenário 1: SleepEEG → Epilepsy
	@$(MAKE) train-full PRETRAIN_DATASET=SleepEEG TARGET_DATASET=Epilepsy

.PHONY: scenario-har
scenario-har: ## Cenário 2: HAR → Gesture (recomendado para GPU 4GB)
	@$(MAKE) train-full PRETRAIN_DATASET=HAR TARGET_DATASET=Gesture

.PHONY: scenario-fd
scenario-fd: ## Cenário 3: FD_A → FD_B (requer GPU 8GB+)
	@$(MAKE) train-full PRETRAIN_DATASET=FD_A TARGET_DATASET=FD_B

.PHONY: scenario-ecg
scenario-ecg: ## Cenário 4: ECG → EMG
	@$(MAKE) train-full PRETRAIN_DATASET=ECG TARGET_DATASET=EMG

# ==============================================================================
# Utilitários
# ==============================================================================

.PHONY: shell
shell: ## Acessar shell interativo no container (com GPU)
	@echo "$(GREEN)🐚 Abrindo shell interativo...$(NC)"
	docker compose --profile shell run --rm tfc-shell

.PHONY: shell-cpu
shell-cpu: ## Acessar shell interativo no container (sem GPU)
	@echo "$(GREEN)🐚 Abrindo shell interativo (CPU)...$(NC)"
	docker compose --profile cpu run --rm --entrypoint /bin/bash tfc-cpu

.PHONY: logs
logs: ## Ver logs do container
	docker compose logs -f

.PHONY: status
status: ## Verificar status dos containers
	docker compose ps -a

.PHONY: check-gpu
check-gpu: build ## Verificar se GPU está disponível no container
	@echo "$(GREEN)🔍 Verificando GPU no container...$(NC)"
	docker run --rm --gpus all $(IMAGE_NAME):$(IMAGE_TAG) \
		python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ==============================================================================
# Limpeza
# ==============================================================================

.PHONY: clean
clean: ## Remover containers parados
	@echo "$(YELLOW)🧹 Removendo containers parados...$(NC)"
	docker compose down --remove-orphans

.PHONY: clean-all
clean-all: clean ## Remover imagem e containers
	@echo "$(YELLOW)🧹 Removendo imagem Docker...$(NC)"
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true

.PHONY: clean-logs
clean-logs: ## Limpar logs de experimentos
	@echo "$(YELLOW)🧹 Limpando logs de experimentos...$(NC)"
	rm -rf code/experiments_logs/*

# ==============================================================================
# Ajuda
# ==============================================================================

.PHONY: help
help: ## Mostrar esta ajuda
	@echo "$(GREEN)TFC-pretraining - Comandos disponíveis:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Variáveis configuráveis:$(NC)"
	@echo "  PRETRAIN_DATASET  Dataset fonte (default: HAR)"
	@echo "  TARGET_DATASET    Dataset alvo (default: Gesture)"
	@echo "  DEVICE            cuda ou cpu (default: cuda)"
	@echo "  SEED              Seed para reprodutibilidade (default: 42)"
	@echo ""
	@echo "$(GREEN)Exemplos:$(NC)"
	@echo "  make build                    # Construir imagem"
	@echo "  make download-data            # Baixar datasets"
	@echo "  make pretrain                 # Pré-treino HAR→Gesture"
	@echo "  make finetune                 # Fine-tuning HAR→Gesture"
	@echo "  make scenario-eeg             # Pipeline SleepEEG→Epilepsy"
	@echo "  make pretrain PRETRAIN_DATASET=ECG TARGET_DATASET=EMG"

.DEFAULT_GOAL := help
