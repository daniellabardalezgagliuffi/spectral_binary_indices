def plotIndexIndex(xtag, ytag, indexdata, polyvertices, xplotrange, yplotrange, **fit_kwargs):
    
    #**fit_kwargs: xfitrange = two-element array with the limits of x fitting region
    
    import os
    import time
    import numpy as np
    import splat
    from numpy.polynomial.polynomial import polyval
    import matplotlib.pyplot as plt
    import matplotlib.path as path
    from matplotlib import rc
    import seaborn as sns
    font = {'family' : 'serif', 'serif':[], 'weight' : 'bold', 'size'   : 16}
    rc('xtick', labelsize=16)
    rc('ytick', labelsize=16)
    rc('font', **font)
    rc('text', usetex=True)
    sns.set_style('white')
    sns.set_context('poster')


    xypoints = np.zeros((len(indexdata),2))
    xypoints[:,1] = indexdata[ytag]
    if xtag == 'Spectral Type':
        xypoints[:,0] = indexdata[xtag].map(lambda x: splat.typeToNum(x))
    else:
        xypoints[:,0] = indexdata[xtag]
    
    #plot known SB
    xfitrange = fit_kwargs.get('xfitrange', None)
    coeffs = fit_kwargs.get('coeffs', None)
    
    if 'xfitrange' in fit_kwargs:
        xarr = np.linspace(xfitrange[0],xfitrange[1],num=50)
        polycurve = polyval(xarr,coeffs)
        curvepts = [[xarr[x],polycurve[x]] for x in range(len(polycurve))]
        flatten = lambda l: [item for sublist in l for item in sublist]
        polyverts = flatten([curvepts,polyvertices])
        polyvertices = polyverts

    p = path.Path(polyvertices)
    pts = p.contains_points(xypoints)
    inpoly = np.where(pts == True)[0]

    fig = plt.figure()
    if len(xypoints) > 1000:
        plt.plot(xypoints[:,0],xypoints[:,1],'.',alpha=0.05)
    else:
        plt.plot(xypoints[:,0],xypoints[:,1],'.')
    plt.xlim(xplotrange)
    plt.ylim(yplotrange)
    plt.xlabel(xtag, fontsize=18)
    plt.ylabel(ytag, fontsize=18)
    if xtag == 'Spectral Type':
        xspt =  np.arange(6)*2+18
        xlabels = ['M8','L0','L2','L4','L6','L8']
        plt.xticks(xspt,xlabels)

    for i in range(len(polyvertices)):
        if i == len(polyvertices)-1:
            plt.plot([polyvertices[i][0],polyvertices[0][0]],[polyvertices[i][1],polyvertices[0][1]],'r')
        else:
            plt.plot([polyvertices[i][0],polyvertices[i+1][0]],[polyvertices[i][1],polyvertices[i+1][1]],'r')

    outputfile = fit_kwargs.get('outputfile', None)
    if 'outputfile' in fit_kwargs:
        plt.savefig(outputfile+'.eps')

    #end = time.time()
    #print(end-start)
    
    return inpoly, fig
#    return inpoly


#################################

def plotIndexSet(indexdf,indexset,outputfile):
    
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import time
    import os
    
    #start = time.time()
    
    #newdir = r'/home/dbardale/python/binarypopsims_30may2017/'+outputfile+'/'
    #if not os.path.exists(newdir):
    #    os.makedirs(newdir)
    
    paramsdf = pd.read_pickle('/Users/daniella/Research/M7L5Sample/BinaryPopSimulations/bg18params.pickle')
    
    def multipage(filename, figs=None, dpi=500):
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()
    
    
    table = np.zeros((len(indexdf),12))
    ind = []
    header = []
    
    if indexset == 'bardalez':
        for i in range(12):
            head = paramsdf['xtag'][i]+'_vs_'+paramsdf['ytag'][i]
            header = np.append(header,head)
            if i < 7:
                # add fig to ii,fig = plotIndexIndex
                ii,fig = plotIndexIndex(paramsdf['xtag'][i],paramsdf['ytag'][i],indexdf,paramsdf['polyvertices'][i],paramsdf['xplotrange'][i],paramsdf['yplotrange'][i])
            if i >= 7:
                ii,fig = plotIndexIndex(paramsdf['xtag'][i],paramsdf['ytag'][i],indexdf,paramsdf['polyvertices'][i],paramsdf['xplotrange'][i],paramsdf['yplotrange'][i],xfitrange=paramsdf['xfit'][i],coeffs=paramsdf['coeffs'][i])
            ind.append(ii)

    figs = [plt.figure(n) for n in plt.get_fignums()]
    plt.close()

    multipage('indexSet_'+outputfile+'.pdf', figs, dpi=250)
    
    candidates = pd.DataFrame(0,index=np.arange(len(indexdf)),columns=header)
    
    for i in range(12):
        candidates[header[i]].ix[ind[i]] = 1
    
    candidates['Total'] = [candidates.ix[i].sum() for i in range(len(indexdf))]
    
    candidates.to_csv('BG14cand_tally_'+outputfile+'.csv')
    df = pd.concat([indexdf,candidates],axis=1)
    df.to_csv('propsbinindexcand_'+outputfile+'.csv')

    # end = time.time()
    # print(end-start)

    return candidates
