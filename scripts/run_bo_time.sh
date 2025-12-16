#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJ_ROOT="$SCRIPT_DIR/.."

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Conda command not found in PATH."
fi

cd $PROJ_ROOT
conda activate ml-sched

python src/main.py --config-name=config_time \
    n_trials=100

cd $SCRIPT_DIR
conda deactivate
