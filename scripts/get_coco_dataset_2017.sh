#!/bin/bash

### Recommended to run 'nohup ./<this_script> &' to prevent interruption from SSH session termination.

wait_to_finish() {
    for pid in "${download_pids[@]}"; do
        while kill -0 "$pid"; do
            sleep 30
        done
    done
}


# Update for default OS specific package manager.
# sudo yum -y install java-1.8.0
# sudo yum -y remove java-1.7.0-openjdk

mkdir -p coco/images/ coco/annotations/

download_pids=()

### 2017 COCO Dataset ###

echo "Downloading COCO dataset..."
curl -OL "http://images.cocodataset.org/zips/val2017.zip" &
download_pids+=("$!")
curl -OL "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" &
download_pids+=("$!")

wait_to_finish download_pids

inflate_pids=()

unzip 'val2017.zip' -d coco/images/ &
inflate_pids+=("$!")
unzip 'annotations_trainval2017.zip' -d coco/annotations/ & # Inflates to 'coco/annotations'.
inflate_pids+=("$!")

wait_to_finish inflate_pids