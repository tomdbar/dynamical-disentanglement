#!/bin/bash
dldir="_data/"

# Build urls
url="https://storage.googleapis.com/3d-shapes/3dshapes.h5"
file="${url##*/}"

### Check for dir, if not found create it using the mkdir ##
[ ! -d "$dldir" ] && mkdir -p "$dldir"

echo Downloading 3dshapes.h5 from "$url" to "${dldir}/${file}"...

# Now download it
wget --progress=bar --show-progress -qc "$url" -O "${dldir}/${file}"

echo done!