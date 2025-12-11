# ------------------------------------------------------------
# Script PowerShell: Ejecución Automática (Ubicación: src)
# ------------------------------------------------------------

# Obtener la ruta donde está guardado este script (.ps1)
$SCRIPT_DIR = $PSScriptRoot

# Nombres de archivos .py (Se asume que están en la misma carpeta que este script)
$FILE_GESTURE_GLOBAL = "eval_gesture.py"
$FILE_GESTURE_ISOLATED = "eval_isolated_gesture.py"
$FILE_PATHOLOGY = "eval_pathology_global.py"

# --- FUNCIÓN GIT ---
function Run-GitCommit ($TaskName) {
    Write-Host "`n--------------------------------------------------" -ForegroundColor Cyan
    Write-Host "GIT: Guardando resultados para $TaskName..." -ForegroundColor Cyan
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan
    
    git add .
    git commit -m "docs(auto): Resultados generados para $TaskName"
    git push
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK: Git Push exitoso." -ForegroundColor Green
    } else {
        Write-Host "Warning: Git Push no realizo cambios o fallo, continuamos..." -ForegroundColor Yellow
    }
}

# --- INICIO DEL PROCESO ---
Write-Host "==============================================" -ForegroundColor Magenta
Write-Host " INICIANDO EVALUACIONES (DESDE $SCRIPT_DIR) " -ForegroundColor Magenta
Write-Host "==============================================" -ForegroundColor Magenta

# Asegurarnos de que estamos en el directorio del script
Set-Location -Path $SCRIPT_DIR

# ----------------------------------
# 1. EVALUACIÓN DE GESTOS (GLOBAL)
# ----------------------------------
Write-Host "`n[1/3] Ejecutando: $FILE_GESTURE_GLOBAL" -ForegroundColor White

if (Test-Path $FILE_GESTURE_GLOBAL) {
    py $FILE_GESTURE_GLOBAL
    Run-GitCommit "Clasificacion Global de Gestos"
} else {
    Write-Error "No se encuentra el archivo: $FILE_GESTURE_GLOBAL"
}

# ----------------------------------
# 2. EVALUACIÓN DE GESTOS AISLADOS (Loop 0-4)
# ----------------------------------
Write-Host "`n[2/3] Ejecutando: $FILE_GESTURE_ISOLATED (Iterando Gestos 0-4)" -ForegroundColor White

if (Test-Path $FILE_GESTURE_ISOLATED) {
    # Bucle para probar cada gesto
    0..4 | ForEach-Object {
        $g_id = $_
        Write-Host "   -> Evaluando Gesto ID: $g_id" -ForegroundColor Yellow
        py $FILE_GESTURE_ISOLATED --gesture-id $g_id
    }
    # Un solo commit al final de los 5 gestos
    Run-GitCommit "Clasificacion de Gestos Aislados (G0-G4)"
} else {
    Write-Error "No se encuentra el archivo: $FILE_GESTURE_ISOLATED"
}

# ----------------------------------
# 3. EVALUACIÓN DE PATOLOGÍA (GLOBAL)
# ----------------------------------
Write-Host "`n[3/3] Ejecutando: $FILE_PATHOLOGY" -ForegroundColor White

if (Test-Path $FILE_PATHOLOGY) {
    py $FILE_PATHOLOGY
    Run-GitCommit "Clasificacion de Patologias"
} else {
    Write-Error "No se encuentra el archivo: $FILE_PATHOLOGY"
}

# ----------------------------------
# FIN
# ----------------------------------
Write-Host "`n==============================================" -ForegroundColor Green
Write-Host " LISTO. QUE DESCANSES. " -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green