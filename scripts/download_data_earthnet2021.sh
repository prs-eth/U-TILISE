#!/usr/bin/env bash


function download_data_earthnet2021() {
    if [ ! -d "data" ]; then
        mkdir -p "data"
    fi
    cd data

    url="https://share.phys.ethz.ch/~pf/stuckercdata/u-tilise/data/"

    data_file="earthnet2021_iid.tar.gz"

    wget --no-check-certificate --show-progress "$url$data_file"
    tar -xzf "$data_file"
    rm "$data_file"
    cd ../
}

download_data_earthnet2021;
