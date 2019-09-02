#!/bin/bash
img_dir="Large-Training-Data"
img_dir_len=${#img_dir}+3
save_dir="raw_im_all"
FILES=$(ls ./$img_dir/*.JPG)

for f in $FILES
do
    ff=${f::${#f}-4}
    echo "$f"
    echo "${ff:${img_dir_len}}"
    convert $f -crop 10x10@ +repage +adjoin ./$save_dir/${ff:${img_dir_len}}_10_%d.JPG
done
