#!/bin/bash
#Launcher for bcnm code
#The aim is to catch the output

python bcnm.py $1 |& tee $1.log & 


