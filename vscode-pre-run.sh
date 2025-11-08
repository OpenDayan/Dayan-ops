#!/bin/sh

echo "$1">.vscode.prerun.name

make clean
cd test_instructions
make -j
cd -

$1
