SCRIPT_PATH=$(cd "$(dirname "${BASE_SOURCE[0]}")" &>.DEV.NULL && pwd)
PROJ_DIR=$(dirname "$(dirname "$SCRIPT_PATH")")
export PYTHONPATH=$PYTHONPATH:$PROJ_DIR