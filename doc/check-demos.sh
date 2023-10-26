#! /bin/bash

# Check that all demos are working
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
echo "Script directory: $SCRIPT_DIR"
echo "$SCRIPT_DIR/tutorials/pyxem-demos"
if [ -d "$SCRIPT_DIR/tutorials/pyxem-demos" ]; then
		echo "The doc/tutorials/pyxem-demos folder does exist"
	else
		echo "The ./tutorials/pyxem-demos folder does not exist"
		echo "Fetching the pyxem-demos from the repository"
		git clone https://github.com/pyxem/pyxem-demos.git "$SCRIPT_DIR/tutorials/pyxem-demos"
	fi