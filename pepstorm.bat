@echo off
set root_dir=%~dp0
pushd %root_dir%
cd tests
for /D %%G in ("test_*") do (
    cd %%G
	for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
	cd ..
)
cd ../pyxem
for %%G in (components generators libraries signals utils) do (
	cd %%G
	for /R %%G in ("*.py") do autopep8 --aggressive --in-place --max-line-length 130 %%G
	cd ..
)
popd
