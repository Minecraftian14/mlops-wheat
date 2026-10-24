@echo off

if "%~1"=="--help" goto :help_message
if "%~1"=="-h"     goto :help_message
if "%~1"=="/?"     goto :help_message

echo Initializing environment for development.

call :check_python_setup
call :activate_python_environment
call :add_scripts_to_path
call xonsh
exit /b

:: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

:check_python_setup

    REM Check if the `uv` command exists, if not print error and quit.
    where uv >nul 2>nul
    if not errorlevel 1 goto :skip_uv_not_found_report
    echo uv command was not found!
    echo Install at https://docs.astral.sh/uv/getting-started/installation
    exit /b 1
    :skip_uv_not_found_report

    REM Check if `.venv` folder exists, if not call on `uv venv` and then `uv sync`.
    if exist .venv goto :skip_venv_creation
    call uv venv || ( echo Creating of Python Virtual Environment failed! & exit /b 1 )
    call uv sync --all-extras --all-groups || ( echo Python Virtual Environment sync failed! & exit /b 1 )
    :skip_venv_creation

    exit /b

:: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

:activate_python_environment
    call .venv\Scripts\activate || ( echo "Unable to activate Python Virtual Environment!" & exit /b )
    exit /b

:: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

:add_scripts_to_path
    where run_tests >nul 2>nul
    if not errorlevel 1 goto :skip_path_update
    set "PATH=%~dp0src\scripts;%PATH%"
    :skip_path_update
    exit /b

:: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

:help_message
    echo.
    echo This script sets up and starts the development environment.
    echo.
    echo The following steps are performed:
    echo   1. Create the Python virtual environment if .venv does not exist,
    echo      then sync its dependencies.
    echo   2. Activate the Python virtual environment.
    echo   3. Add the project's scripts directory to PATH.
    echo   4. Start xonsh.
    echo.
    echo Requirements:
    echo   uv must be available on PATH.
    echo.
    echo Usage:
    echo   init          Run the setup steps and start xonsh.
    echo   init --help   Print this message.
    echo   init -h       Print this message.
    echo   init /?       Print this message.
    echo.
    echo Note:
    echo   After running init,
    echo    - Run initialize_env to initialize the .env file.
    echo    - Run rename_project to update the project name.
    echo.
    exit /b
