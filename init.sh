#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

main() {
    if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "/?" ]; then
        help_message
        return 0
    fi

    echo "Initializing environment for development."

    check_python_setup || return 1
    activate_python_environment || return 1
    add_scripts_to_path
    xonsh
}

# :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

check_python_setup() {
    # Check if the `uv` command exists, if not print error and quit.
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv command was not found!"
        echo "Install at https://docs.astral.sh/uv/getting-started/installation"
        return 1
    fi

    # Check if `.venv` folder exists, if not call on `uv venv` and then `uv sync`.
    if [ ! -d ".venv" ]; then
        uv venv || { echo "Creating of Python Virtual Environment failed!"; return 1; }
        uv sync --all-extras --all-groups || { echo "Python Virtual Environment sync failed!"; return 1; }
    fi
}

# :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

activate_python_environment() {
    if [ -f ".venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate || { echo "Unable to activate Python Virtual Environment!"; return 1; }
    elif [ -f ".venv/Scripts/activate" ]; then
        # shellcheck disable=SC1091
        source .venv/Scripts/activate || { echo "Unable to activate Python Virtual Environment!"; return 1; }
    else
        echo "Unable to activate Python Virtual Environment!"
        return 1
    fi
}

# :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

add_scripts_to_path() {
    if ! command -v run_tests >/dev/null 2>&1; then
        export PATH="$SCRIPT_DIR/src/scripts:$PATH"
    fi
}

# :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

help_message() {
    echo ""
    echo "This script sets up and starts the development environment."
    echo ""
    echo "The following steps are performed:"
    echo "  1. Create the Python virtual environment if .venv does not exist,"
    echo "     then sync its dependencies."
    echo "  2. Activate the Python virtual environment."
    echo "  3. Add the project's scripts directory to PATH."
    echo "  4. Start xonsh."
    echo ""
    echo "Requirements:"
    echo "  uv must be available on PATH."
    echo ""
    echo "Usage:"
    echo "  init.sh          Run the setup steps and start xonsh."
    echo "  init.sh --help   Print this message."
    echo "  init.sh -h       Print this message."
    echo "  init.sh /?       Print this message."
    echo ""
    echo "Note:"
    echo "  After running init,"
    echo "   - Run initialize_env to initialize the .env file."
    echo "   - Run rename_project to update the project name."
    echo ""
}

# :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: :: ::

main "$@"
