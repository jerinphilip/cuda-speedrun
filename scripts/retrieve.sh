#!/bin/bash

set -x

URL=http://www.cse.iitm.ac.in/~rupesh/teaching/gpu/jan23/work/

WGET_OPTIONS=(
  --recursive
  --no-parent     #
  --convert-links # Convert links for local viewing.
  --no-host-directories
)

wget "${WGET_OPTIONS[@]}" \
  --directory-prefix tmp/ \
  $URL

mv tmp/~rupesh/teaching/gpu/jan23/work iitm-jan23-gpu
