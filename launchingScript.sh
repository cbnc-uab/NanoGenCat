#!/bin/bash
#Launcher for bcnm code
#The aim is to catch the output

/usr/bin/python3.6 bcnm.py $1 | tee $1.log & 


