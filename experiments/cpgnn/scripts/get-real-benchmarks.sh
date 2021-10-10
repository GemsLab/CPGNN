#!/usr/bin/env bash
# LINK=https://drive.google.com/file/d/1JV5AG_B2-XYL-YAI8Hwk3PScWh_uR53P/view?usp=sharing
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1JV5AG_B2-XYL-YAI8Hwk3PScWh_uR53P'
ggURL='https://drive.google.com/uc?export=download'
TARGET=archives/real-benchmarks.tar.gz
SHA256SUM=ed108c51b01dfd039eda90a6449eb463e60da5bd50ce07418b0d3fdc342e0617

cd "$(dirname ${BASH_SOURCE[0]})/.."
mkdir -p archives

# Automatic downloading script adopted from https://stackoverflow.com/a/38937732
filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r
tar -xvzf $TARGET
