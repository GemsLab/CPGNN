#!/usr/bin/env bash
LINK=https://umich.box.com/shared/static/b1f1s6gf1heh3dy4jqqvxp947vtlz0yl.gz
TARGET=archives/real-benchmarks.tar.gz
SHA256SUM=ed108c51b01dfd039eda90a6449eb463e60da5bd50ce07418b0d3fdc342e0617

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives
wget -L $LINK -O $TARGET
echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
