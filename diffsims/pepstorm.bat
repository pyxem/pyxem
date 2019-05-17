@echo off
set root_dir=%~dp0
pushd %root_dir%
cd pyxem/tests
for /D %%G in ("test_*") do (
    cd %%G
	for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
	cd ..
)
cd ../
for %%G in (components generators libraries signals utils) do (
	cd %%G
	for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
	cd ..
)
popd
