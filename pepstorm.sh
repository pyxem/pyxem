#!/bin/bash
cd "$(dirname "$0")"
cd pyxem/tests
for folder in test_components test_generators test_library test_physical test_signals test_utils
	do
	cd $folder
	autopep8 *.py --aggressive --in-place --max-line-length 130
	cd .. 
done
cd ../
for folder in components generators io_plugins libraries signals utils  
	do 
	cd $folder
	autopep8 *.py --aggressive --in-place --max-line-length 130
	cd ..
done
 
