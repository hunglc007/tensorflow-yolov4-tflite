#!/usr/bin/env bash


PARENT_DIR="$( cd "$( dirname $( dirname $( dirname "${BASH_SOURCE[0]}" ) ) )" >/dev/null 2>&1 && pwd )"
DATA_DIR="$PARENT_DIR/data"

DATASET_NAME="VOCtrainval_11-May-2012"

wget -c -P $DATA_DIR http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

if [[ -d "$DATA_DIR/$DATASET_NAME" ]]; then
    echo "Already '$DATA_DIR/$DATASET_NAME' path exists."
    exit 1
fi

mkdir $DATA_DIR/$DATASET_NAME

tar xf $DATA_DIR/VOCtrainval_11-May-2012.tar -C $DATA_DIR/$DATASET_NAME

