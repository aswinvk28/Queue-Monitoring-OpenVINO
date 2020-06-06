#!/bin/bash

# TODO: Create DEVICE variable
DEVICE=$1
# TODO: Create MODEL variable
MODEL=$2
# TODO: Create VIDEO variable
VIDEO=$3
# TODO: Create QUEUE variable
QUEUE=$4
OUTPUT=$5
# TODO: Create PEOPLE variable
PEOPLE=$6
# mkdir -p $OUTPUT

python queue_app.py  --model ${MODEL} \
                          --device ${DEVICE} \
                          --video ${VIDEO} \
                          --output ${OUTPUT} \
                          --queue_param ${QUEUE} \
                          --max_people ${PEOPLE} \
                          --confidence_level 0.65 \

# cd /output

# tar zcvf output.tgz *