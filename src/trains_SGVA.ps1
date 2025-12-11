# ------------------------------------------------------------
# Script PowerShell: train base (Healthy) + finetune (Patologías) + git push
# ------------------------------------------------------------

# Variables de Configuración
$EPOCHS_BASE = 100
$EPOCHS_FINETUNE = 100
$BATCH_SIZE = 256
$SEED = 0

# Variables del Modelo Condicional
$C_DIM = 7 
$MAX_BETA_FT = 3.0 


# Rutas de guardado
$TRAIN_BASE_PATH = "./saved_model/train_base"
$FINETUNE_CVAE_PATH = "./saved_model/finetune_cVAE"

# Ruta de la carpeta del proyecto para Git
$GIT_PROJECT_PATH = "C:\Users\Awild\OneDrive\Desktop\SGVA-CV"

# ----------------------------------
# FASE 1: Entrenamiento Base (Healthy Only) - Paso 2 del Plan
# ----------------------------------
Write-Host "=============================================="
Write-Host "FASE 1: Entrenamiento Base (Healthy Only) - Paso 2"
Write-Host "=============================================="

# Definición de argumentos de la FASE 1 como una lista de strings (SIN BACKTICKS)
$ArgsPhase1 = @(
    "--epochs", $EPOCHS_BASE,
    "--batch-size", $BATCH_SIZE,
    "--seed", $SEED,
    "--c-dim", $C_DIM,
    "--ft-mode", "base",
    "--outpath", $TRAIN_BASE_PATH
)

# Ejecución de Python
py train_diva.py $ArgsPhase1


# Ruta del mejor modelo preentrenado (para la FASE 2)
$PRETRAINED_MODEL = "$TRAIN_BASE_PATH/diva_best_seed$SEED.model"

# Verificar que el modelo se haya guardado
if (-not (Test-Path $PRETRAINED_MODEL)) {
    Write-Error "ERROR: No se encontró el modelo pre-entrenado en $PRETRAINED_MODEL. Deteniendo el script."
    exit 1
}

# ----------------------------------
# FASE 2: Fine-Tuning Condicional (Healthy + Patologías) - Paso 3 del Plan
# ----------------------------------
Write-Host "=============================================="
Write-Host "FASE 2: Fine-Tuning Condicional (Patologías) - Paso 3"
Write-Host "=============================================="

# Definición de argumentos de la FASE 2 como una lista de strings (SIN BACKTICKS)
$ArgsPhase2 = @(
    "--epochs", $EPOCHS_FINETUNE,
    "--batch-size", $BATCH_SIZE,
    "--seed", $SEED,
    "--c-dim", $C_DIM,
    "--max_beta", $MAX_BETA_FT,
    "--pretrained-model", "$PRETRAINED_MODEL",
    "--ft-mode", "finetune",
    "--outpath", $FINETUNE_CVAE_PATH
)

# Ejecución de Python
py train_diva.py $ArgsPhase2
    
# ----------------------------------
# Git add + commit + push
# ----------------------------------
Write-Host "=============================================="
Write-Host "Committing and pushing changes to Git..."
Write-Host "=============================================="

Set-Location -Path $GIT_PROJECT_PATH
git add .
git commit -m "feat: Entrenamiento base y fine-tuning condicional (Patologías) completados."
git push

Write-Host "Script completado. Modelos finales guardados en $FINETUNE_CVAE_PATH"
