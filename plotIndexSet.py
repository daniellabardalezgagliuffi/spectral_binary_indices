def plotIndexSet(paramsdf,indexdf,indexset):
    
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import pandas as pd
    
    def multipage(filename, figs=None, dpi=200):
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
                ii,fig = plotIndexIndex(paramsdf['xtag'][i],paramsdf['ytag'][i],indexdf,paramsdf['polyvertices'][i],paramsdf['xplotrange'][i],paramsdf['yplotrange'][i])
            if i >= 7: 
                ii,fig = plotIndexIndex(paramsdf['xtag'][i],paramsdf['ytag'][i],indexdf,paramsdf['polyvertices'][i],paramsdf['xplotrange'][i],paramsdf['yplotrange'][i],xfitrange=paramsdf['xfit'][i],coeffs=paramsdf['coeffs'][i])
            ind.append(ii)
        figs = [plt.figure(n) for n in plt.get_fignums()]
        
        multipage('indexSet.pdf', figs, dpi=250)
        
        candidates = pd.DataFrame(index=np.arange(len(indexdf)),columns=header)
        
        for i in range(len(indexdf)):
            candidates[header[i]][ind[i]] = 1

        candidates['Total'] = [candidates.ix[i].count() for i in range(len(indexdf))]
        
        candidates.to_csv('BG14spectral_indices_tally.csv')
        
    return candidates
