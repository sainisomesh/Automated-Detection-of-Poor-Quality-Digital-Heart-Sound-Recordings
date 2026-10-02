# Shared setup for run_all.sh and revision/run_all.sh. Source, don't execute.
#
# Finds a Python 3.9-3.13 interpreter, creates a virtual environment in
# reproducibility/.venv (unless one is already active), and puts it first on
# PATH so every later `python` call uses it. Works in bash on Linux, macOS,
# and Windows (Git Bash or WSL).

REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLBACKEND=Agg
export PYTHONUTF8=1

# ask "question" -> returns 0 for yes. Defaults to no. With ASSUME_YES=1 it
# answers yes without prompting; with no terminal attached it answers no.
ask() {
    if [ "${ASSUME_YES:-0}" = "1" ]; then
        echo "$1 [y/N]: y (--yes)"
        return 0
    fi
    local reply=""
    if [ -t 0 ]; then
        read -r -p "$1 [y/N]: " reply || reply=""
    else
        echo "$1 [y/N]: n (no terminal)"
    fi
    case "$reply" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

find_python() {
    local cmd
    for cmd in python3 python "py -3"; do
        if $cmd -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) else 1)' >/dev/null 2>&1; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

setup_python() {
    if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
        if ! python -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 13) else 1)' >/dev/null 2>&1; then
            echo "ERROR: the active environment's Python must be 3.9-3.13." >&2
            exit 1
        fi
        echo "Using the active environment: $(python -c 'import sys; print(sys.executable)')"
        return
    fi

    local venv="$REPRO_ROOT/.venv"
    if [ ! -d "$venv" ]; then
        local py
        py="$(find_python)" || {
            echo "ERROR: Python 3.9-3.13 is required (found none on PATH as python3, python or py)." >&2
            exit 1
        }
        echo "Creating virtual environment in $venv using: $py"
        $py -m venv "$venv"
    fi
    if [ -d "$venv/Scripts" ]; then
        export PATH="$venv/Scripts:$PATH"
    else
        export PATH="$venv/bin:$PATH"
    fi
    export VIRTUAL_ENV="$venv"
    echo "Using $(python -c 'import sys; print(sys.executable)')"
}

install_requirements() {
    python -m pip install --upgrade pip --quiet
    python -m pip install -r "$1"
}
