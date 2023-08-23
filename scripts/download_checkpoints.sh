#!/usr/bin/env bash


function download_checkpoints() {
    if [ ! -d "checkpoints" ]; then
        mkdir -p "checkpoints"
    fi

    url="https://share.phys.ethz.ch/~pf/stuckercdata/u-tilise/checkpoints/"

    # Get the list of all .pth files
    pth_files=$(wget -qO- "$url" | grep -oE 'href="[^"]+\.pth"' | cut -d'"' -f2)

    # Download each .pth file to the local directory 'checkpoints'
    for file in $pth_files; do
        wget --no-check-certificate --show-progress -O checkpoints/"$file" "$url$file"
    done
}

download_checkpoints;
