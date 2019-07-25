#false positive rate: how many single objects are picked up by this technique.
def calculateIndices(bindf):
    
    import numpy as np
    import splat
    import pandas as pd
    
    spind = []
    for i in np.arange(len(bindf)):
        print(i)
        if pd.notnull(bindf['binsp'][i]):
            tmpind = splat.measureIndexSet(bindf['binsp'][i], set='bardalez')
            print(tmpind)
        else:
            tmpind = np.nan
            print(tmpind)
        spind.append(tmpind)

    tags = list(spind[0].keys())
    indexdf = pd.DataFrame(columns=[tags],index=np.arange(len(bindf)))
    for i in range(len(bindf)):
        if pd.notnull(spind[i]):
            indexdf.loc[i] = np.array(list(spind[i].values()))[:,0]
        else:
            indexdf.loc[i] = np.zeros(len(spind[0]))*np.nan

    indexdf['Spectral Type'] = bindf['Spectral Type']
    
    return indexdf
