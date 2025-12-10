# ------------------------------------------------------------
# Script PowerShell: train + finetune + git push
# ------------------------------------------------------------

# Variables
$EPOCHS_TRAIN = 100
$EPOCHS_FINETUNE = 100
$BATCH_SIZE = 256
$SEED = 0

# Rutas de guardado
$TRAIN_PATH = "./saved_model/train"
$FINETUNE_PATH = "./saved_model/finetune"

# ----------------------------------
# Entrenamiento normal
# ----------------------------------
Write-Host "Starting normal training..."
python train_diva.py `
    --epochs $EPOCHS_TRAIN `
    --batch-size $BATCH_SIZE `
    --seed $SEED `
    --outpath $TRAIN_PATH

# ----------------------------------
# Fine-tuning
# ----------------------------------
# Tomamos el modelo preentrenado del entrenamiento normal
$PRETRAINED_MODEL = "$TRAIN_PATH/diva_best_seed$SEED.model"

Write-Host "Starting fine-tuning..."
python train_diva.py `
    --epochs $EPOCHS_FINETUNE `
    --batch-size $BATCH_SIZE `
    --seed $SEED `
    --pretrained-model $PRETRAINED_MODEL `
    --outpath $FINETUNE_PATH

# ----------------------------------
# Git add + commit + push
# ----------------------------------
Write-Host "Committing and pushing changes..."
Set-Location -Path "C:\Users\Awild\OneDrive\Desktop\SGVA-CV"
git add .
git commit -m "Modelos entrenados y fine-tune realizados"
git push




# ./train.ps1