@echo off
REM Windows launcher: installs the environment with uv, builds the model if missing, starts the app.
cd /d "%~dp0"

python -m uv --version >nul 2>nul
if errorlevel 1 (
    echo Installing uv for this user...
    python -m pip install --user uv || goto :error
)

python -m uv sync || goto :error

if not exist "models\model.pkl" (
    echo No model found - trying DagsHub first, then rebuilding locally ^(about 10 minutes^)...
    python -m uv run dvc pull || python -m uv run dvc repro || goto :error
)

python -m uv run streamlit run app\streamlit_app.py --server.port 8501
pause
exit /b 0

:error
echo Setup failed. See the messages above.
pause
exit /b 1
