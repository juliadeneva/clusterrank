import sys,os
from subprocess import *
from glob import glob

if len(sys.argv) != 2:
    print 'Usage: driver.py [topdir]'
    print '(Topdir is the parent of all beam dirs for a given drift strip)'
    sys.exit(1)

topdir = sys.argv[1]

if not os.path.exists(topdir):
    print 'Directory '+topdir+' does not exist!'
    sys.exit(1)

os.chdir(topdir)
dirs = glob('*D*')
dirs.sort()

n = len(dirs)
ii = 1
for d in dirs:
    print '*** Dir %d of %d ***' %(ii,n)
    # Have to do this to go around (pylab?) memory leaks    
    pr = Popen('python /home/deneva/python/clusterrank/spplot.py '+d,shell=True,stdin=PIPE)
    pr.wait()
    del pr

    ii = ii+1

# Move all plots to a separate dir
#if not os.path.exists('histplots'):
#    os.mkdir('histplots')
#pr = Popen('mv D*/hist*.png histplots/.',shell=True,stdin=PIPE)
#pr.wait()

spdir = 'spplots'

if not os.path.exists(spdir):
    os.mkdir(spdir)
pr = Popen('mv *D*/*.png '+spdir+'/.',shell=True,stdin=PIPE)
pr.wait()

# Merge and sort histrank lists, and move to plot dir
pr = Popen('cat *D*/histrank.txt | sort -n -r -k 4 > '+spdir+'/histrank-all.txt',shell=True,stdin=PIPE)
pr.wait()




