# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 16:20:57 2017

@author: Stefan
"""

import numpy as np

data = np.genfromtxt('TCQ_timetags.dat', dtype=None)

tmp = 0
i = 0
counter = 0
counter62 = 0
counter26 = 0
for val in data:
    if (tmp == val[0]):
        print str(i) + " " + str(data[i-1][0]) + " " + str(data[i-1][1])
        print str(i) + " " + str(val[0]) + " " + str(val[1])
        print
        counter = counter + 2
        if (data[i-1][1] == 6):
            counter62 = counter62 + 2
        else:
            counter26 = counter26 + 2
    tmp = val[0]
    i = i+1

print
print "double events: " + str(counter)   
print "62 Events: " + str(counter62)
print "26 Events: " + str(counter26)  