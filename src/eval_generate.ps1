# --- CONFIGURACIÓN ---
# Ruta de tu proyecto (Asegúrate de que esta sea correcta)
$GIT_PROJECT_PATH = "C:\Users\Awild\OneDrive\Desktop\SGVA-CV"

# Nombres de los archivos (Ajustados a los que te pasé recién)
$FILE_FIDELITY = "eval_fidelity.py"
$FILE_GESTURE_GLOBAL = "eval_gestures_global.py"
$FILE_GESTURE_ISOLATED = "eval_gestures_isolated.py"
$FILE_PATHOLOGY = "eval_pathology.py"

# --- FUNCIÓN GIT ---
function Run-GitCommit ($TaskName) {
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan
    Write-Host "GIT: Guardando resultados para $TaskName..." -ForegroundColor Cyan
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan
    
    git add .
    git commit -m "docs(auto): Resultados de evaluación generados para $TaskName"
    git push
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Git Push exitoso." -ForegroundColor Green
    } else {
        Write-Host "⚠️ Error en Git Push, pero continuamos..." -ForegroundColor Yellow
    }
}

# Nos movemos al directorio del proyecto
Set-Location -Path $GIT_PROJECT_PATH

Write-Host "==============================================" -ForegroundColor Magenta
Write-Host " INICIANDO SECUENCIA DE EVALUACIÓN AUTOMÁTICA " -ForegroundColor Magenta
Write-Host "==============================================" -ForegroundColor Magenta

# ----------------------------------
# 2. EVALUACIÓN DE GESTOS (GLOBAL)
# ----------------------------------
Write-Host "`n[2/4] Ejecutando: $FILE_GESTURE_GLOBAL" -ForegroundColor White
py $FILE_GESTURE_GLOBAL

# Commit Fase 2
Run-GitCommit "Clasificación Global de Gestos"

# ----------------------------------
# 3. EVALUACIÓN DE GESTOS AISLADOS (Loop 0 a 4)
# ----------------------------------
Write-Host "`n[3/4] Ejecutando: $FILE_GESTURE_ISOLATED (Iterando Gestos 0-4)" -ForegroundColor White

# Bucle automático para probar cada gesto por separado
0..4 | ForEach-Object {
    $g_id = $_
    Write-Host "   -> Evaluando Gesto ID: $g_id" -ForegroundColor Yellow
    # Llamamos al script pasando el argumento del gesto
    py $FILE_GESTURE_ISOLATED --gesture-id $g_id
}

# Commit Fase 3 (Se hace uno solo al terminar los 5 gestos para no saturar el repo)
Run-GitCommit "Clasificación de Gestos Aislados (G0-G4)"

# ----------------------------------
# 4. EVALUACIÓN DE PATOLOGÍA (GLOBAL)
# ----------------------------------
Write-Host "`n[4/4] Ejecutando: $FILE_PATHOLOGY" -ForegroundColor White
py $FILE_PATHOLOGY


# Commit Fase 4
Run-GitCommit "Clasificación de Patologías"

# ----------------------------------
# 1. EVALUACIÓN DE FIDELIDAD (C2ST)
# ----------------------------------
Write-Host "`n[1/4] Ejecutando: $FILE_FIDELITY" -ForegroundColor White
py $FILE_FIDELITY

# Commit Fase 1
Run-GitCommit "Fidelidad (Real vs Generated)"


# ----------------------------------
# FIN
# ----------------------------------
Write-Host "`n==============================================" -ForegroundColor Green
Write-Host " TAREA COMPLETADA. ¡BUENOS DÍAS! ☕ " -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green