#!/usr/bin/env bash

wget -O simlocmatch-dataset.tar.xz https://imperialcollegelondon.box.com/shared/static/48w09bdd5n763qdp7ph5352cwnfqaan4.xz
echo "Extracting tar file"
mkdir -p data
tar -xvf simlocmatch-dataset.tar.xz
echo "Deleting tar file"
rm simlocmatch-dataset.tar.xz
echo "All done."
