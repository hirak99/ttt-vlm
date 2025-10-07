#!/bin/bash

set -uexo pipefail

readonly MY_PATH="$(dirname "$(realpath "$0")")"
cd "$MY_PATH"/src

streamlit run gui_play.py

