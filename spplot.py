import sys, os
from glob import glob
import numpy as np
from pylab import *
from math import pi,exp
from scipy.special import erf
from scipy.optimize import leastsq
from time import time,clock

# 2 --> interactive plots, pause for keypress; print memory usage and cpu time
# 1 --> plots saved as png, print memory usage and cpu time
# 0 --> plots saved as png
debug = 0
# min number of events to consider a cluster
mincluster = 50 
# max acceptable gap in time between events in a cluster 
#maxtgap = 0.01 
# sample time * max nsmooth (150) in Presto SP detection
maxtgap = 0.018
# max acceptable gap in dm between events in a cluster 
maxdmgap = 1.0 
# how many iterations to use for finding outliers (more is better but slower; 0 means don't find and reject outliers; 3-5 iterations works well)
nfitlevel = 3 

# For AO327 (Mock)
fcenter = 0.327 #GHz center frequency
bw = 57.0 #MHz bandwidth

if debug > 0:
    import resource
    def Using(point):
        usage=resource.getrusage(resource.RUSAGE_SELF)
        return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )

def globfiles(matchthis):
    # Argument is a list of strings
    files = []
    for m in matchthis:
        files = files + glob(m)

    return files

# Classic SP search diagnostic plot
def spplot_classic(times,dms,sigmas,fname,fig):

    figure(fig.number)

    # Number of pulses vs DM
    ax1 = subplot2grid((2,2),(0,0))
    hist(dms, bins=len(flist)/2)
    xlabel('DM(pc/cc')
    ylabel('Number of pulses')

    # SNR vs DM
    ax2 = subplot2grid((2,2),(0,1))
    plot(dms,sigmas,'ko',markersize=1)
    xlabel('DM(pc/cc)')
    ylabel('SNR')

    # DM vs time plot
    ax3 = subplot2grid((2,2),(1,0),colspan=2)
    autoscale(enable=True, axis='both', tight=True)
    plot(times[(sigmas<=6)], dms[(sigmas<=6)],'bo',mfc='None',markersize=1)
    plot(times[np.logical_and(sigmas>6,sigmas<=10)], dms[np.logical_and(sigmas>6, sigmas<=10)],'bo',mfc='None',markersize=2)
    plot(times[np.logical_and(sigmas>10,sigmas<=20)], dms[np.logical_and(sigmas>10, sigmas<=20)],'bo',mfc='None',markersize=5)
    plot(times[(sigmas>20)], dms[(sigmas>20)],'bo',mfc='None',markersize=10)
    xlabel('Time(s)')
    ylabel('DM(pc/cc)')

    suptitle(plottitle, fontsize=12)    
    if debug > 1:
        show()
    else:
        fig.savefig(plottitle+'.png',bbox_inches=0)
        fig.clf()

def plotcluster(times,dms,sigmas,n):
    colors = ['ro','go','bo','co']
    ncolors = len(colors)
    noccolor = 'ko' # for events not part of a cluster

    if len(times) >= mincluster:
        c = colors[n%ncolors]
    else:
        c = noccolor

    plot(times[(sigmas<=6)], dms[(sigmas<=6)],c,mec='none',markersize=2)
    plot(times[np.logical_and(sigmas>6,sigmas<=10)], dms[np.logical_and(sigmas>6, sigmas<=10)],c,mec='none',markersize=4)
    plot(times[np.logical_and(sigmas>10,sigmas<=20)], dms[np.logical_and(sigmas>10, sigmas<=20)],c,mec='none',markersize=7)
    plot(times[(sigmas>20)], dms[(sigmas>20)],c,mec='none',markersize=10)

def residuals(p,y,x):
    psrdm,psrw,psrsigma = p
    zeta = 6.91e-3 * bw * (x-psrdm) / (psrw * fcenter*fcenter*fcenter)

    err = zeros(zeta.shape)
    ii = (abs(zeta) < 1e-20)
    # For instances where zeta is zero use L'Hopital's rule 
    err[ii] = y[ii] - psrsigma
    # Now calculate the actual value for the rest
    err[~ii] = y[~ii] - psrsigma * 0.5*sqrt(pi) / zeta[~ii] * erf(zeta[~ii])

    # This is just as slow:
    # For instances where zeta is zero use L'Hopital's rule 
    #err = y - psrsigma
    # Now calculate the actual value for the rest
    #ii = find(abs(zeta) > 1e-20)
    #err[ii] = y[ii] - psrsigma * 0.5*sqrt(pi) / zeta[ii] * erf(zeta[ii])

    return err

def peval(x,p):
    zeta = 6.91e-3 * bw * (x-p[0]) / (p[1] * fcenter*fcenter*fcenter)
    return p[2] * 0.5*sqrt(pi) / zeta * erf(zeta)

def fitcluster(times,dms,sigmas,n):
    dmpenalty = 1.0
    colors = ['ro','go','bo','co']
    ncolors = len(colors)
    c = colors[n%ncolors]

    imax = np.argmax(sigmas)
    bestdm = dms[imax]
    bestsigma = sigmas[imax]
    
    if debug > 1:
        plot(dms,sigmas,c,mec='none',label='Good points')
        plot([bestdm,bestdm],[min(sigmas),max(sigmas)],'k-')

    # Starting guesses for the pulsar DM, W(ms), and sigma
    p0 = [0.99*bestdm,1,bestsigma]
    # Initial fit
    plsq = leastsq(residuals, p0, args=(sigmas, dms),full_output=1,ftol=1.e-04,xtol=1.e-04)

    for ifit in range(0,nfitlevel):
        sigma_exp = peval(dms,plsq[0])
        if ifit == 0 and debug > 1:
            plot(dms,sigma_exp,'k--',label='Initial fit')

        # Find and plot outliers, with more stringent limits at the tails
        if debug > 1:
            ii = find(abs(sigmas-sigma_exp) >= 0.5*abs(sigma_exp-5))
            #plot(dms[ii],sigmas[ii],'ko')
        iigood = find(abs(sigmas-sigma_exp) < 0.5*abs(sigma_exp-5))
    
        # Redo fit without outliers
        niigood = len(iigood)
        if niigood == 0:
            imax = np.argmax(sigmas)
            return 0.0,times[imax],dms[imax]
        else:
            goodsigmas = sigmas[iigood]
            gooddms = dms[iigood]
            if len(goodsigmas) < 3:
                imax = np.argmax(goodsigmas)
                return 0.0,times[imax],dms[imax]
            else:
                plsq = leastsq(residuals, plsq[0], args=(goodsigmas, gooddms),full_output=1,ftol=1.e-04,xtol=1.e-04)
   
        # Plot final fit if outlier rejection ran
        if ifit == nfitlevel-1 and debug > 1:
            ii = find(abs(sigmas-sigma_exp) >= 0.5*abs(sigma_exp-5))
            plot(dms[ii],sigmas[ii],'ko',label='Outliers')
            sigma_exp = peval(gooddms,plsq[0])
            plot(gooddms,sigma_exp,'k-',label='Final fit')

    # Plot final fit if there was no outlier rejection, and calculate R^2
    if nfitlevel == 0:
        ss_tot=((sigmas-sigmas.mean())**2).sum()
        imax = np.argmax(sigmas)
        t = times[imax]
        dm = dms[imax]
        sigma_exp = peval(dms,plsq[0])
        plot(dms,sigma_exp,'k-',label='Final fit')
    else:
        ss_tot=((goodsigmas-goodsigmas.mean())**2).sum()
        imax = np.argmax(goodsigmas)
        t = (times[iigood])[imax]
        dm = gooddms[imax]
        
    ss_err=((plsq[2])['fvec']**2).sum()
    rsquared=1.0-(ss_err/ss_tot)
    
    # If the best dm or pulse width fitted are negative, this is not a real pulse, so peg test statistic to zero
    if plsq[0][0] < 0.0 or plsq[0][1] < 0.0:
        rsquared = 0

    # Penalize RFI even if SNR vs DM shape gives a good fit
    if dm < dmpenalty:
        rsquared = max([rsquared-exp(-dm*dm),0.0])

    if debug > 1 :
        xlabel('DM(pc/cc)')
        ylabel('SNR')
        suptitle('R^2: %4.2f at t = %3.2f s' % (rsquared,t))
        legend(bbox_to_anchor=(0.7, 0.8, 0.3, 0.2), loc=1, mode="expand", borderaxespad=0.,frameon=False)
        #legend(bbox_to_anchor=(0, 0, 1, 1), loc=1,bbox_transform=gcf().transFigure)

    return rsquared,t,dm

def spplot_clusters(times,dms,sigmas,plottitle,fig1,fig2):    

    if debug > 0:
        t0 = time()
        c0 = clock()

    n = len(times)
    nplotted = 0
    
    if debug > 1:
        ion() #turn on interactive mode for paging through plots

    #fig1 = figure() # for classic SP plot, with color clusters
    figure(fig1.number)

    # Number of pulses vs DM
    ax1 = subplot2grid((2,2),(0,0))
    hist(dms, bins=len(flist)/2)
    xlabel('DM(pc/cc')
    ylabel('Number of pulses')

    # SNR vs DM
    ax2 = subplot2grid((2,2),(0,1))
    plot(dms,sigmas,'ko',markersize=1)
    xlabel('DM(pc/cc)')
    ylabel('SNR')

    # DM vs time
    ax3 = subplot2grid((2,2),(1,0),colspan=2)
    autoscale(enable=True, axis='both', tight=True)

    if debug > 1:
        #fig2 = figure() # for cluster by cluster paging
        figure(fig2.number)

    # Sort all events by arrival
    isort = np.argsort(times) 

    icluster = []
    nclusters = 0
    bestts = 0.0
    bestt = 0.0
    bestdm = 0.0
    for ii in range(0,n):
        if len(icluster) == 0: #starting a new cluster
            icluster.append(isort[ii]) 

        if ii < n-1:
            ilow = isort[ii]
            ihigh = isort[ii+1]
            dt = times[ihigh]-times[ilow]

        if ii==n-1 or dt > maxtgap:
            # Plot cluster
            ctimes = times[icluster]
            cdms = dms[icluster]
            csigmas = sigmas[icluster]
            len0 = len(ctimes)
            len1 = 0

            # Sort events close in time by dm
            nj = len(ctimes)
            jsort = np.argsort(cdms)
            jcluster = []

            for jj in range(0,nj):
                if len(jcluster) == 0: #starting a new cluster
                    jcluster.append(jsort[jj])

                if jj < nj-1:
                    jlow = jsort[jj]
                    jhigh = jsort[jj+1]
                    ddm = cdms[jhigh]-cdms[jlow]

                if jj==nj-1 or ddm > maxdmgap:
                    cctimes = ctimes[jcluster]
                    ccdms = cdms[jcluster]
                    ccsigmas = csigmas[jcluster]
                    len1 = len1 + len(cctimes)
                    if len(cctimes) > mincluster:
                        nclusters = nclusters + 1

                        # Do SNR vs DM fitting for each cluster
                        if debug > 1:
                            figure(fig2.number)
                        ts,t,dm = fitcluster(cctimes,ccdms,ccsigmas,nclusters)
                        if ts > bestts:
                            bestts = ts
                            bestt = t
                            bestdm = dm
                        # For paging through individual cluster plots
                        if debug > 1:
                            raw_input()
                            fig2.clf()

                    # DM vs time plot
                    figure(fig1.number)
                    plotcluster(cctimes,ccdms,ccsigmas,nclusters)
                    nplotted = nplotted + len(cctimes)

                    jcluster = []
                else:
                    jcluster.append(jsort[jj+1])

            #if len0 != len1:
            #    print 'Lost events! Time cluster: %d Total from DM sub-clusters: %d\n' %(len0,len1)
            #    sys.exit()

            icluster=[]

        else:
            icluster.append(isort[ii+1])

    print 'Ntotal: %d Nplotted: %d Clusters: %d Best R^2: %4.2f' % (n,nplotted,nclusters,bestts)
    
    figure(fig1.number)
    xlabel('Time(s)')
    ylabel('DM(pc/cc)')    
    suptitle('%s\nBest R^2: %4.2f at t = %3.2f, DM = %4.2f' % (plottitle,bestts,bestt,bestdm), fontsize=12)
    
    if debug > 1:
        raw_input()
    else:
        fig1.savefig('%1.2f%s%s%s' % (bestts,'_',plottitle,'.png'),bbox_inches=0)
        fig1.clf()

    if debug > 0:
        t1 = time()
        c1 = clock()
        print "spplot_clusters Wall time: %e s" % (t1-t0)
        print "spplot_clusters CPU time: %e s" % (c1-c0)

    return bestts 

#########################################################################

if __name__ == "__main__":
    
    # DM = 0 - 39
    tmp1 = ['*DM[0-9].*singlepulse','*DM[1-3]?.*singlepulse']
    # DM = 30 - 119
    tmp2 = ['*DM[3-9]?.*singlepulse','*DM1[01]?.*singlepulse']
    # DM = 100 - 319
    tmp3 = ['*DM[12]??.*singlepulse', '*DM3[01]?.*singlepulse']
    # DM = 300 - 519
    tmp4 = ['*DM[34]??.*singlepulse', '*DM5[01]?.*singlepulse']
    # DM = 500 - 1999 (search tops out at ~1100)
    tmp5 = ['*DM[5-9]??.*singlepulse', '*DM1???.??.*singlepulse']
    # DM = 30 - 300
    tmp6 = ['*DM[3-9]?.*singlepulse', '*DM[1-3]??.*singlepulse']
    # DM = 100 - 500
    tmp7 = ['*DM[1-4]??.*singlepulse']
    # DM = 300 - 1999
    tmp8 = ['*DM[3-9]??.*singlepulse', '*DM1???.??.*singlepulse']
    # All DMs
    tmp9 = ['*.singlepulse']

    dmlists = [tmp1,tmp2,tmp7,tmp5]
    #dmlists = [tmp2]

    if len(sys.argv) != 2:
        print 'Usage: spplot.py [beamdir]'
        sys.exit(1)

    beamdir = sys.argv[1]

    if not os.path.exists(beamdir):
        print 'Directory '+beamdir+' does not exist!'
        sys.exit(1)

    os.chdir(beamdir)

    if debug > 0:
        t0 = time()
        c0 = clock()

    # See if reusing one figure instance will prevent memory leakage-->doesn't
    fig1 = figure()
    fig2 = figure()

    listcount = 0
    for dmlist in dmlists:
        flist = globfiles(dmlist)

        if len(flist) == 0:
            print 'No DM files in current glob: '+' '.join(dmlist)
            listcount = listcount+1
            continue

        plottitle = flist[0].split('_')[0]+'_SP'+str(listcount)
        if len(glob('*'+plottitle+'.png')) > 0:
            print 'File already made: '+plottitle+'.png'
            listcount = listcount+1
            continue

        print '\nWorking on '+plottitle

        times = []
        dms = []
        sigmas = []
        for f in flist:
            events = np.loadtxt(f,usecols=(0,1,2),dtype={'names': ('dm', 'sigma', 'time'), 'formats': ('f4', 'f4', 'f4')})
            
            # If there's only one event in the file, it ends up in a numpy '0-d array', which has to be handled differently
            if len(events['time'].shape) == 0: 
                times = np.append(times,events['time'])
                dms = np.append(dms,events['dm'])
                sigmas = np.append(sigmas,events['sigma'])
            else:
                times = np.concatenate((times,events['time']))
                dms = np.concatenate((dms,events['dm']))
                sigmas = np.concatenate((sigmas,events['sigma']))    

        fig1.clf()
        fig2.clf()

        # What causes infs to be recorded for event SNR?
        iinf = find(sigmas == inf)
        if len(iinf) > 0:
            print 'Found sigma == inf! No cluster fitting.'
            spplot_classic(times,dms,sigmas, plottitle,fig1)
        else:
            bestts = spplot_clusters(times,dms,sigmas,plottitle,fig1,fig2)
    
        listcount = listcount + 1
    
        if debug > 0:
            print Using('')

            t1 = time()
            c1 = clock()

            print "spplot_main Wall time: %e s" % (t1-t0)
            print "spplot_main CPU time: %e s\n" % (c1-c0)

