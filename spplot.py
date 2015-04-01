import sys, os
from glob import glob
import numpy as np
from pylab import *
from math import pi,exp,log10
from scipy.special import erf
from scipy.optimize import leastsq
from time import time,clock
from scipy.stats import kurtosis
#import matplotlib.cm as cm

# 2 --> interactive plots, pause for keypress; print memory usage and cpu time
# 1 --> plots saved as png, print memory usage and cpu time
# 0 --> plots saved as png
debug = 2
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
# fractional tolerance for fit
tol = 1.e-04
# If this is True, divide the best R^2 for the beam by log10(nclusters) to rank down beams with lots of RFI
docorr = True

# For AO327 (Mock)
fcenter = 0.327 #GHz center frequency
bw = 57.0 #MHz bandwidth
print '\n****** AO327 (MOCK) ******'

# For LWA1
#fcenter = 0.076 #GHz center frequency
#bw = 19.6 #MHz bandwidth
#print '\n****** LWA1 ******'

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
    xlabel('DM(pc/cc)')
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

def plotcluster(times,dms,sigmas,n,ts):
    colors = ['go','bo','co']
    ncolors = len(colors)
    noccolor = 'ko' # for events not part of a cluster

    #noccolor = [0., 0., 0., 1.] 
    #colors = cm.jet(np.linspace(0, 1, 11))
    
    if len(times) >= mincluster:
        if ts > 0.8:
            c = 'ro'
        elif ts > 0.7 and ts < 0.8:
            c = 'mo'
        elif ts > 0.6 and ts < 0.7:
            c = 'co'
        elif ts > 0.5 and ts < 0.6:
            c = 'go'
        else:
            c = 'bo'
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
    colors = ['go','bo','co']
    ncolors = len(colors)
    c = colors[n%ncolors]
    #c = 'ro'

    subplot(2,1,2)
    plot(dms,times,c,mec='none',label='Good points')
    ylabel('Time (s)')
    xlabel('DM(pc/cc)')
    
    subplot(2,1,1)

    imax = np.argmax(sigmas)
    bestdm = dms[imax]
    bestsigma = sigmas[imax]
    
    if debug > 1:
        plot(dms,sigmas,c,mec='none',label='Good points')
        plot([bestdm,bestdm],[min(sigmas),max(sigmas)],'k-')

    # Starting guesses for the pulsar DM, W(ms), and sigma
    #p0 = [0.99*bestdm,1,bestsigma]
    p0 = [0.99*bestdm,10,bestsigma]
    # Initial fit
    plsq = leastsq(residuals, p0, args=(sigmas, dms),full_output=1,ftol=tol,xtol=tol)

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
                plsq = leastsq(residuals, plsq[0], args=(goodsigmas, gooddms),full_output=1,ftol=tol,xtol=tol)
   
        # Plot final fit if outlier rejection ran
        if ifit == nfitlevel-1 and debug > 1:
            ii = find(abs(sigmas-sigma_exp) >= 0.5*abs(sigma_exp-5))
            plot(dms[ii],sigmas[ii],'ko',label='Outliers')
            sigma_exp = peval(gooddms,plsq[0])
            plot(gooddms,sigma_exp,'k-',label='Final fit')

            subplot(2,1,2)
            plot(dms[ii],times[ii],'ko',label='Outliers')

            subplot(2,1,1)

    # Plot final fit if there was no outlier rejection, and calculate R^2
    if nfitlevel == 0:
        ss_tot=((sigmas-sigmas.mean())**2).sum()
        imax = np.argmax(sigmas)
        t = times[imax]
        #dm = dms[imax]
        sigma_exp = peval(dms,plsq[0])
        plot(dms,sigma_exp,'k-',label='Final fit')
    else:
        ss_tot=((goodsigmas-goodsigmas.mean())**2).sum()
        imax = np.argmax(goodsigmas)
        t = (times[iigood])[imax]
        #dm = gooddms[imax]
    
    dm = plsq[0][0]
        
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
        suptitle('R^2: %4.2f at t = %3.2f s, DM = %3.2f pc/cc' % (rsquared,t,dm))
        legend(bbox_to_anchor=(0.7, 0.8, 0.3, 0.2), loc=1, mode="expand", borderaxespad=0.,frameon=False)
        #legend(bbox_to_anchor=(0, 0, 1, 1), loc=1,bbox_transform=gcf().transFigure)

    return rsquared,t,dm

def spplot_clusters(times,dms,sigmas,plottitle,fig1,fig2):    

    if debug > 0:
        t0 = time()
        c0 = clock()

    f = open('clusterrank-t-dm-r2.txt','a')

    n = len(times)
    nplotted = 0
    
    if debug > 1:
        ion() #turn on interactive mode for paging through plots

    #fig1 = figure() # for classic SP plot, with color clusters
    figure(fig1.number)
    subplots_adjust(top=0.87)

    # Number of pulses vs DM
    ax1 = subplot2grid((2,2),(0,0))
    hist(dms, bins=len(flist)/2)
    xlabel('DM(pc/cc)')
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
            ts = 0.0

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
                        f.write('%10.4f  %7.2f  %6.2f\n' % (t,dm,ts))
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
                    plotcluster(cctimes,ccdms,ccsigmas,nclusters,ts)
                    nplotted = nplotted + len(cctimes)
                    ts = 0.0

                    jcluster = []
                else:
                    jcluster.append(jsort[jj+1])

            #if len0 != len1:
            #    print 'Lost events! Time cluster: %d Total from DM sub-clusters: %d\n' %(len0,len1)
            #    sys.exit()

            icluster=[]

        else:
            icluster.append(isort[ii+1])

    # Close pulselist file
    f.close()

    # Penalize beams with many clusters, because those tend to be due to RFI
    if nclusters > 0 and docorr:
        tscorrection = round(log10(nclusters))
        print 'bestts correction: %f' % tscorrection
        if tscorrection > 1:
            bestts = bestts/tscorrection

    print 'Ntotal: %d Nplotted: %d Clusters: %d Best R^2: %4.2f' % (n,nplotted,nclusters,bestts)
    
    figure(fig1.number)
    xlabel('Time(s)')
    ylabel('DM(pc/cc)')    
    suptitle('%s\nClusters: %d Best R^2: %4.2f at t = %3.2f, DM = %4.2f\n  ' % (plottitle,nclusters,bestts,bestt,bestdm), fontsize=11)
    
    if debug > 1:
        raw_input()
    #else:
    #    fig1.savefig('%1.2f%s%s%s' % (bestts,'_',plottitle,'.png'),bbox_inches=0)
    
    if debug > 0:
        t1 = time()
        c1 = clock()
        print "spplot_clusters Wall time: %e s" % (t1-t0)
        print "spplot_clusters CPU time: %e s" % (c1-c0)
    
    return bestts 

# Find valleys between peaks of a histogram
# (the thresh arg refers to threshold for peakiness for saving vA,P,vB sets)
def findvals(h,thresh):
    valleys = []
    peaks = []
    peakinesses = []
    nbins = len(h)
    #print nbins
    
    j = 0
    vA = h[j] # initialize 1st valley
    P = vA # initialize peak
    vB = 0 # initialize 2nd valley

    #The width of the valley, default width is 1
    W = 1

    #The sum of the counts between vA and vB
    N = 0

    #The measure of the peak's peakiness
    peakiness = 0.0

    peak = 0
    l = False

    try:
        while j < nbins:
            l = False
            vA = h[j]
            P = vA
            W = 1
            N = vA

            i = j + 1

            #To find the peak
            while P < h[i]:
                P = h[i]
                W = W + 1
                N = N + h[i]
                i = i + 1

            #To find the border of the valley other side
            peak = i - 1
            vB = h[i]
            N = N + h[i]
            i = i + 1
            W = W + 1

            l = True
            while vB >= h[i]:
                vB = h[i]
                W = W + 1
                N = N + h[i]
                i = i + 1

            #Calculate peakiness
            peakiness = (1.0 - (vA + vB) / (2.0 * P)) * (1 - N/(W * P))
                        
            #if (peakiness > thresh) and (j not in valleys):
            if peakiness > thresh:
                peaks.append(peak)
                peakinesses.append(peakiness)
                valleys.append(j)
                valleys.append(i - 1)

            j = i - 1
    except:
        if (l):
            vB = h[-1]

            peakiness = (1.0 - (vA + vB) / (2.0 * P)) * (1 - N/(W * P))
            
            if peakiness > thresh:
                valleys.append(j)
                valleys.append(nbins-1)
                peaks.append(nbins-1)
                peakinesses.append(peakiness)
            
            return valleys,peaks,peakinesses
        
    return valleys,peaks,peakinesses

# Look for peaks in the histogram of # of pulses vs DM
def histrank(dms,plottitle,fig):
    if debug > 1:
        figure(fig.number)
        fig.clf()

    #n, bins, patches = hist(dms,100)
    n,bins = np.histogram(dms,100)
    # get the bin centers (hist returns edges)
    bincenters = 0.5*(bins[1:]+bins[:-1])

    #plot(bincenters,n,'k')
    
    # Smooth the histogram (not good, makes peaks less sharp)
    #nsmooth = floor(len(n)/10.)
    #nsmooth = 2.0
    #wts = np.repeat(1.0, nsmooth) / nsmooth
    #histsm = np.convolve(wts,n,mode='same')
    histsm = n

    if debug > 1:
        plot(bincenters,histsm,'k')
    
    # Find valleys - here N_val = 2 * N_pks
    vals,pks,pkns = findvals(histsm,0.5)

    # Find valleys - here N_val = N_pks + 1
    #vals = (diff(sign(diff(histsm))) > 0).nonzero()[0] + 1 # local min
    #pks = (diff(sign(diff(histsm))) < 0).nonzero()[0] + 1 # local max
    #vals = np.concatenate(([0],vals))
    #vals = np.concatenate((vals,[len(histsm)-1]))

    nmax = max(n)
    nmin = min(n)

    if debug > 1:
        for v in vals:
            plot([bincenters[v],bincenters[v]],[nmin,nmax],'b-')
        for p in pks:
            plot([bincenters[p],bincenters[p]],[nmin,nmax],'r-')
    
    nvals = len(vals)
    npks = len(pks)
    
    # Possible histts (histrank merit figures):
    # [0]:Peakiness * Nevents in highest bin of peak
    # [1]:Peakiness * Nevents within peak
    # [2]:Kurtosis of peak * Nevents in highest bin of peak
    # [3]:Kurtosis of peak * Nevents within peak
    histts = [-9999,-9999,-9999,-9999]
    bestdms = [-9999,-9999,-9999,-9999]

    for ii in range(0,npks):
        # values belonging to current peak
        a = histsm[vals[2*ii]:vals[2*ii+1]] # if using findvals
        #a = histsm[vals[ii]:vals[ii+1]]
        kurt = kurtosis(a)
        currdm = bincenters[pks[ii]]

        # exclude peaks close to DM=0
        if currdm < 1.5:
            continue

        tmp = pkns[ii]*histsm[pks[ii]]
        if histts[0] < tmp:
            histts[0] = tmp
            bestdms[0] = currdm
        tmp = pkns[ii]*sum(a)
        if histts[1] < tmp:
            histts[1] = tmp
            bestdms[1] = currdm
        tmp = kurt*histsm[pks[ii]]
        if histts[2] < tmp:
            histts[2] = tmp
            bestdms[2] = currdm
        tmp = kurt*sum(a)
        if histts[3] < tmp:
            histts[3] = tmp
            bestdms[3] = currdm
        #print 'DM: %6.2f Pkns: %3.2f Kurt: %5.2f Pkns*Nmax: %6.2f Pkns*Npk: %6.2f Kurt*Nmax: %6.2f Kurt*Npk: %6.2f' % (bincenters[pks[ii]],pkns[ii],kurt,pkns[ii]*histsm[pks[ii]],pkns[ii]*sum(a),kurt*histsm[pks[ii]],kurt*sum(a))
    
    if debug> 1:
        suptitle('Pkns*Nmax: %6.1f (DM=%6.1f) Pkns*Npk: %6.1f (DM=%6.1f)\n Kurt*Nmax: %6.1f (DM=%6.1f) Kurt*Npk: %6.1f (DM=%6.1f)' % (histts[0],bestdms[0],histts[1],bestdms[1],histts[2],bestdms[2],histts[3],bestdms[3]))
        xlabel('DM (pc/cc)')
        ylabel('# of pulses')
        raw_input()

    return histts,bestdms

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

    #dmlists = [tmp1,tmp2,tmp7,tmp5]
    dmlists = [tmp1]

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

    if debug > 1:
        ion() #turn on interactive mode for paging through plots

    # See if reusing one figure instance will prevent memory leakage-->doesn't
    fig1 = figure()
    fig2 = figure()
    fig3 = figure()

    # Text output for histrank
    histout = open('histrank.txt','a')

    listcount = 0
    for dmlist in dmlists:
        flist = globfiles(dmlist)

        if len(flist) == 0:
            print 'No DM files in current glob: '+' '.join(dmlist)
            listcount = listcount+1
            continue

        plottitle = flist[0].split('_')[0]+'_SP'+str(listcount)
        if len(glob('*'+plottitle+'.png')) > 0:
            print 'File(s) already made: *'+plottitle+'.png'
            listcount = listcount+1
            continue

        print '\nWorking on '+plottitle

        times = []
        dms = []
        sigmas = []
        for f in flist:
            if os.path.getsize(f) == 0:
                #print 'File %s empty, skipping' % (f)
                continue

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
        fig3.clf()

        # What causes infs to be recorded for event SNR?
        iinf = find(sigmas == inf)
        if len(iinf) > 0:
            print 'Found sigma == inf! No ranking.'
            spplot_classic(times,dms,sigmas, plottitle,fig1)
        else:
            bestts = spplot_clusters(times,dms,sigmas,plottitle,fig1,fig2)
            
            histts,bestdms = histrank(dms,plottitle,fig3)
            histout.write('%6.1f %6.1f %6.1f %6.1f %s\n' % (histts[0],histts[1],histts[2],histts[3],plottitle))

            figure(fig1.number)
            suptitle('\n\n\nPkns*Nmax: %d (DM=%d) Pkns*Npk: %d (DM=%d) Kurt*Nmax: %d (DM=%d) Kurt*Npk: %d (DM=%d)' % (histts[0],bestdms[0],histts[1],bestdms[1],histts[2],bestdms[2],histts[3],bestdms[3]),fontsize=11)

            if debug > 1:
                raw_input()
            else:
                fig1.savefig('%1.2f%s%s%s' % (bestts,'_',plottitle,'.png'),bbox_inches=0)

        listcount = listcount + 1
    
        if debug > 0:
            print Using('')

            t1 = time()
            c1 = clock()

            print "spplot_main Wall time: %e s" % (t1-t0)
            print "spplot_main CPU time: %e s\n" % (c1-c0)

    # Close text output for histrank
    histout.close()
        
