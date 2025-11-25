#!/bin/sh

echo "BUILD STARTING"

echo "$1">.vscode.prerun.name

make clean
cd test_instructions
make -j
cd -
echo "BUILD COMPLETE"

$1 | tee .rpt
