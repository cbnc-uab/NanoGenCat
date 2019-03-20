#!/bin/bash
#Launcher for bcnm_anatase code
#The aim is to catch the output

python bcnm_anatase.py $1 |& tee $1.log & 


