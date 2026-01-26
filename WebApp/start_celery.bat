@echo off
REM ==============================================================================
REM Script para iniciar Celery Worker en Windows
REM ==============================================================================

echo.
echo ========================================
echo   INICIANDO CELERY WORKER
echo ========================================
echo.

REM Navegar al directorio de la aplicación web
cd /d "%~dp0"

REM Verificar si existe el entorno virtual
if exist "..\venv\Scripts\activate.bat" (
    echo [INFO] Activando entorno virtual...
    call ..\venv\Scripts\activate.bat
) else (
    echo [ADVERTENCIA] No se encontro entorno virtual en ..\venv
    echo [INFO] Usando Python del sistema
)

REM Verificar que Redis está corriendo
echo [INFO] Verificando conexion a Redis...
python -c "import redis; r=redis.Redis(host='localhost', port=6379); r.ping(); print('[OK] Redis esta corriendo')" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Redis no esta corriendo o no es accesible
    echo [INFO] Inicia Redis antes de ejecutar Celery
    echo [INFO] Comando: redis-server
    echo.
    pause
    exit /b 1
)

echo.
echo [INFO] Iniciando Celery worker...
echo [INFO] Proyecto: rag_project
echo [INFO] Nivel de log: info
echo [INFO] Pool: solo (Windows compatible)
echo.
echo ========================================
echo   WORKER ACTIVO - Presiona Ctrl+C para detener
echo ========================================
echo.

REM Iniciar Celery worker
celery -A rag_project worker --loglevel=info --pool=solo

REM Si Celery falla
if errorlevel 1 (
    echo.
    echo [ERROR] Celery fallo al iniciar
    echo [INFO] Verifica que todas las dependencias esten instaladas
    echo [INFO] Comando: pip install -r requirements.txt
    echo.
    pause
)
