#!/bin/sh

pip3 install torch torchvision >> batch_pip.log
ret=$?
if [ $ret -ne 0 ]; then
     #Handle failure
     #exit if required
    exit;
fi
python3 main.py >> batch_run.log