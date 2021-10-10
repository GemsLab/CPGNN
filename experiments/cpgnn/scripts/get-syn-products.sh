#!/usr/bin/env bash
# LINK=https://drive.google.com/file/d/1oOT7sn8btnfr1-lxeujW1FIjpdCV8Ler/view?usp=sharing
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1oOT7sn8btnfr1-lxeujW1FIjpdCV8Ler'
ggURL='https://drive.google.com/uc?export=download'
TARGET=archives/syn-products.tar.gz
SHA256SUM=446e25cef0053d7d305381224772cfb6ad7a3f1f8e8981b7cb8e86690e30b3e4

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives

# Automatic downloading script adopted from https://stackoverflow.com/a/38937732
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
