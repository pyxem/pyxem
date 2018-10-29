#!/bin/bash
cd "$(dirname "$0")"
ls
cd tests
ls
for folder in test_components test_generators test_library test_physical test_signals test_utils
	do
	cd $folder
	autopep8 *.py --aggressive --in-place --max-line-length 100
	cd .. 
done
cd ../pyxem
for folder in components generators libraries signals utils  
	do 
	cd $folder
	autopep8 *.py --aggressive --in-place --max-line-length 100
	cd ..
done
 
