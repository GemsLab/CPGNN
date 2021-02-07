#!/usr/bin/env bash
LINK=https://umich.box.com/shared/static/69lm6quzrxb8dt2g7bw89akfypx3ozn0.gz
TARGET=archives/syn-products.tar.gz
SHA256SUM=446e25cef0053d7d305381224772cfb6ad7a3f1f8e8981b7cb8e86690e30b3e4

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives
wget -L $LINK -O $TARGET
echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
