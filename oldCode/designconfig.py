#!/usr/bin/python

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util

import random
import time

import zmq
import sys

# Class to facilitate ZMQ communication
class MaxLab:
    def __init__(self, hostname='localhost'):
        self.hostname = hostname
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect ("tcp://" + self.hostname + ":6022")
    def send(self, cmd, parameters=""):
        self.socket.send_string(cmd + " " + parameters)
        r = self.socket.recv()
        print r
        return r

m = MaxLab()

# Input arguments
expDir  = str(sys.argv[1])
recEls  = str(sys.argv[2])
stimEls = str(sys.argv[3])
outFile = str(sys.argv[4])

print "input file:  "+expDir+recEls
print "input file:  "+expDir+stimEls
print "output file: "+expDir+outFile

# Select recording electrodes specified in elFile 
f=open(expDir+recEls, 'r')
temp=f.read().splitlines()
rec_electrodes = [ str(el) for el in temp ]
f.close()
rec_electrodes_list = ' '.join(rec_electrodes)

# Select stimulation electrodes specified in elFile 
f=open(expDir+stimEls, 'r')
temp=f.read().splitlines()
stim_electrodes = [ str(el) for el in temp ]
f.close()
stim_electrodes_list = ' '.join(stim_electrodes)

array = maxlab.chip.Array()
array.reset()
array.clear_selected_electrodes()

array.select_electrodes( [rec_electrodes_list] )
array.select_stimulation_electrodes( [stim_electrodes_list] )
array.route()

print(stim_electrodes)
arr=len(stim_electrodes)
for el_ind in range(0, arr):
	# print(el_ind)
	array.connect_electrode_to_stimulation( stim_electrodes[el_ind] )

myconfig = expDir+outFile
array.save_config(myconfig)





