#! /bin/bash

# Check that all demos are working
if [ -d "doc/tutorials/pyxem-demos" ]; then
		echo "The doc/tutorials/pyxem-demos folder does exist"
	else
		echo "The doc/tutorials/pyxem-demos folder does not exist"
		echo "Fetching the pyxem-demos from the repository"
		(cd "doc/tutorials" || exit
		git clone https://github.com/pyxem/pyxem-demos.git)
	fi