#!/bin/bash

delete_last_created_dir() {
    dir_path=$1
    last_created_dir=$(ls -td -- "$dir_path"/* | head -n 1)
    if [ -n "$last_created_dir" ]; then
        rm -r "$last_created_dir"
        echo "Deleting: $last_created_dir"
    else
        echo "No dir found in $dir_path"
    fi
}

logs_dir="./Logs/Hopper"
checkpoints_dir="./Checkpoints/Hopper"

delete_last_created_dir "$logs_dir"
delete_last_created_dir "$checkpoints_dir"
