#!/bin/bash

set -uexo pipefail

readonly MY_PATH="$(dirname "$(realpath "$0")")"
cd "$MY_PATH"/src

# Default port is first free port, starting at 8501.
streamlit run gui_play.py --server.port 8009
