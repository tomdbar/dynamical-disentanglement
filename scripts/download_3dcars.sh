#!/bin/bash
dldir="_data/"
tmpdir="${dldir}/_tmp"

# Build urls
url="http://www.scottreed.info/files/nips2015-analogy-data.tar.gz"
file="${url##*/}"

### Check for dir, if not found create it using the mkdir ##
[ ! -d "$dldir" ] && mkdir -p "$dldir"
[ ! -d "$tmpdir" ] && mkdir -p "$tmpdir"

echo Downloading 3dshapes.h5 from "$url" to "${tmpdir}/${file}"...
wget  --no-check-certificate --no-proxy --progress=bar --show-progress -qc "$url" -O "${tmpdir}/${file}"

echo unpacking...
tar -C "${tmpdir}" -xzf "${tmpdir}/${file}"
mv "${tmpdir}/data/cars" "${dldir}/3dcars"

echo tidying...
rm -r "${tmpdir}/data"

echo done!