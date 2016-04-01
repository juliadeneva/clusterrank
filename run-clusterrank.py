#!/usr/bin/python

import sys,os
from glob import glob
from subprocess import *

topdir = '/home/deneva/ao327/spplots-puppi'
codedir = '/home/deneva/python/clusterrank'

if len(sys.argv) != 2:
    print 'Usage: run-clusterrank.py [expr]'
    print '(Expr can contain wild cards; describes data dirs to be processed.)'
    sys.exit(1)

datedirs = glob(topdir+'/'+sys.argv[1])
datedirs.sort()
#print datedirs
#sys.exit()

for d in datedirs:
    stripdirs = os.listdir(d)

    for s in stripdirs:
        indir = d+'/'+s
        print indir

        pr = Popen('python '+codedir+'/driver.py '+indir,shell=True,stdin=PIPE)
        pr.wait()
        del pr


