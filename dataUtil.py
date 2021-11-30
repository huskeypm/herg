import matplotlib.pylab as plt
import numpy as np
import pandas as pd


def CalcZScore(df,tag):
    vals = df.loc[df.index,[tag]]
    avg = np.mean(vals)
    stddev = np.std(vals)
    zs = (vals - avg)/stddev
    return zs

# compute relative values
# i don't see the WT data, so well say
#wtName = "Y99S"

def ShapeData(df,tag,wtName,zscore=False,display=False):
    wtDF = df.loc[df['VARIANT'] == wtName]

    # identify trafficking competent
    soluble = df.loc[df['TRAFFICKING'] == 1]
    insoluble = df.loc[df['TRAFFICKING'] == 0]

    # reference each tag by the wt, s.t. more non-traffickers have values larger than wt'
    vWT = np.float(wtDF[tag])

    # figure out which set above/below wt value has more LOF variants
    superThresh = df[df[tag] > vWT]
    nSuperThresh = len(superThresh)
    subThresh = df[df[tag] <= vWT]
    nSubThresh = len(subThresh)
    tot = np.float( len(df) )
    nInSol_sup = len( superThresh[ superThresh['TRAFFICKING']==0] )
    nInSol_sub = len( subThresh[ subThresh['TRAFFICKING']==0] )
    newTag = 'd'+tag

    # reorder such that more LOFs have higher values than wt
    pctSup = nInSol_sup/np.float( nSuperThresh  )
    pctSub = nInSol_sub/np.float( nSubThresh  )

    if (pctSup > pctSub):  # higher values have higher pctg of LOF variants
        df[newTag] = df[tag] - vWT
        label = newTag
    else:
        df[newTag] = vWT - df[tag]    # lower values have higher pctg LOF variants
        label = "-1 x "+newTag
    print(tag,pctSub,pctSup,tot,label)

    if display:
        plt.figure()
        dummy = plotVals(df,tag=newTag,asLog=False,label=label)


    # compute z-scores
    if zscore:
      zs = CalcZScore(df,newTag)
      df.loc[df.index,[newTag]] = zs
      label+=" (z-score)"

      if display:
          plt.figure()
          dummy = plotVals(df,tag=newTag,asLog=False,label=label)


def plotVals(df,tag='dRMSD',asLog=False,label=None):
    # sort
    dfs = df.sort_values(tag)
    l = dfs.reset_index()
    dfs = l
    inds = 1

    # get insol and sol entries
    vals = np.zeros( len(df) )

#    insolInds = dfs.index[dfs['CLASS'] == 1].tolist()
#    insolVs = dfs[dfs['CLASS'] == 1][tag]
#    solInds = dfs.index[dfs['CLASS'] == 0].tolist()
#    solVs = dfs[dfs['CLASS'] == 0][tag]

    insolInds = dfs.index[dfs['TRAFFICKING'] == 0].tolist()
    insolVs = dfs[dfs['TRAFFICKING'] == 0][tag]

    solInds = dfs.index[dfs['TRAFFICKING'] == 1].tolist()
    solVs = dfs[dfs['TRAFFICKING'] == 1][tag]

    # plt
    if asLog:
        insolVs = np.log(insolVs)
        solVs = np.log(solVs)
    plt.bar(insolInds,insolVs,color='red',label='insoluble')
    #plt.bar(solInds,solVs,color='blue',label='sol')
    plt.bar(solInds,solVs,color='green',label='soluble')
    if label is not None:
        plt.ylabel(label)
    plt.title(tag)
    plt.legend()

    return insolVs, solVs

# Calculate true positive probability densities as a function of cutoff
def probCond(df, tag,display=False):
    fileName = tag+"_cumeprob.txt" 

    minVal = np.min( df[tag])
    maxVal = np.max( df[tag])
    iters=20; #print("do more values?")
    threshVals = np.linspace(minVal, maxVal,iters)
    #print(threshVals)
    threshVals = threshVals[:-1] # all but last entry
    probs = np.zeros_like( threshVals )

    for i,threshVal in enumerate( threshVals):
        superThresh = df[df[tag] > threshVal]
        tot = np.float( len(superThresh) )
        #nSol = len( superThresh[ superThresh['CLASS']==0] )
        #nInSol = len( superThresh[ superThresh['CLASS']==1] )

        nSol = len( superThresh[ superThresh['TRAFFICKING']==1] )
        nInSol = len( superThresh[ superThresh['TRAFFICKING']==0] )

        pInSol = nInSol/tot
        probs[i] = pInSol
        #print( threshVal, tot, nSol, nInSol, pInSol )

    # save data 
    stacked = np.stack([threshVals,probs], axis=0)
    np.savetxt(fileName,stacked) 

#    plt.plot(threshVals,probs)
    if display:
        plt.plot(threshVals,probs,label=tag)

        plt.ylabel('Prob(Insoluble|Threshold)')
        plt.xlabel('Threshold')
        plt.legend()
        plt.title("ROC MD") 

    return (threshVals, probs)

#Computes
#$P(Insol | feature > \lambda)$
#and
#$\Pi P(Insol | feature_i > \lambda)$
#
#I've verified that I am computing the correct conditional probability for each feature and variant
#
#
#* Note: adjust the value_variant-value_wt order to make sure all plots are roughly monotonically increasing *
def ProdCondProbs(tagsSubset,df,df_train,display=False):
    plt.figure()
    prods = np.ones(len(df.index))
    for tag in tagsSubset:
    #    plt.figure()
    #    t,p = probCond(df,tag)

        # compute conditional probability based on training data 
        label=tag
        t,p = probCond(df_train,tag,display=display) # train on subset of df

        # interpolate conditional probability based on cond prob from training
        vals = df[tag]
        #print(vals)
        ps = np.interp(vals,t,p) # populate for ALL members of df
        df['pInSol'+tag] = ps
        #print(vals)
        #print(ps)
        prods *= ps

    #df['Prod'] = prods

    #totProd = np.sum(prods)
    maxProd = np.max(prods)
    #print(totProd)
    df['Prod'] = prods/maxProd
    # verify that np.sum(df['Prod']) = 1
    plt.savefig('indi_condprob_md.pdf')




# The product of conditional probabilities gives the likelihood that a member is defective (insoluble), given the products of being insoluble for each feature. We then pick a 'reasonable' threshold to decide if a member is insoluble

def calcRates(df,cutoff=None,display=False,verbose=True):
    insolVs = df[df['TRAFFICKING'] == 0]['Prod']
    solVs = df[df['TRAFFICKING'] == 1]['Prod']

    #cutoff = 0.78 # arbitrary
    allVs = np.concatenate([insolVs,solVs])
    if cutoff is None:
        damax = np.max(allVs)
        damin = np.min(allVs)
        cutoff = np.mean([damin,damax]) # an initial guess

    n = len(allVs)
    if display:
        plt.figure()
        dummy = plotVals(df,tag='Prod')
        plt.plot(np.arange(n), cutoff*np.ones(n))

    insolSupThresh = np.where(insolVs >= cutoff)
    nInsolSupThresh = np.shape(insolSupThresh)[1]
    insolSubThresh = np.where(insolVs < cutoff)
    nInsolSubThresh = np.shape(insolSubThresh)[1]

    solSupThresh = np.where(solVs >= cutoff)
    nsolSupThresh = np.shape(solSupThresh)[1]
    solSubThresh = np.where(solVs< cutoff)
    nsolSubThresh = np.shape(solSubThresh)[1]

    #TPR = nInsolSupThresh/np.float( nInsolSupThresh + nsolSupThresh )
    TPR = nInsolSupThresh/np.float( nInsolSupThresh + nsolSupThresh )
    #FNR = 1 - nInsolSupThresh/np.float( len(insolVs) )
    FNR = np.float( nInsolSubThresh + nsolSubThresh )
    if FNR <= 1e-3:
        FNR = 0.
    else:
        FNR = nInsolSubThresh/FNR

    #FPR = 1 - TPR

    if verbose:
        print("Cutoff ",cutoff)
        print("TPR (insol)", TPR, " Support ", nInsolSupThresh)
        print("FNR (insol)", FNR, " Support ", nInsolSubThresh)
        #print("FPR (insol)", FPR, " Support ", nsolSupThresh)

    return TPR,FNR



# compute ROC curve
def ComputeROC(df, threshVals = 20,display=False):
    prods = df['Prod']
    minThresh = np.min(prods)
    maxThresh = np.max(prods)
    cutoffs = np.linspace(minThresh,maxThresh, threshVals)

    tprs = []
    fnrs = []
    for cutoff in cutoffs:
        tpr,fnr = calcRates(df,cutoff=cutoff,display=False,verbose=False)
        tprs.append(tpr)
        fnrs.append(fnr)

    # add 1,1 point
    tprs.append(1)
    fnrs.append(1)
    tprs = np.asarray(tprs)
    fprs = 1 - tprs
    fnrs = np.asarray(fnrs)
    tnrs = 1 - fnrs

    from sklearn.metrics import auc
    rfnrs = fnrs# + 1e-2*np.arange( len(fnrs ))
    idx = np.argsort(rfnrs)
    rfnrs = np.sort(rfnrs)
    rtprs = tprs[idx]
    daAUC = auc(rfnrs,rtprs)

     
    plt.figure()
    plt.scatter(fnrs,tprs, label="AUC=%4.2f"%daAUC)
    plt.plot(fnrs,tprs)
    xs = np.linspace(0,1,100)
    plt.plot(xs,xs,'--', color="black", label="Random")
    plt.title("ROC (Condition probability, Insoluble)")
    plt.xlabel("FNR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig('roc_conditional_probability_md_features.pdf')


    plt.figure()
    cutoffs = np.concatenate([cutoffs,[1]])
    plt.plot(cutoffs,tprs,'k-',label="tprs") 
    plt.plot(cutoffs,fprs,'r-',label="fprs") 
    plt.plot(cutoffs,tnrs,'k--',label="tnrs") 
    plt.plot(cutoffs,fnrs,'r--',label="fnrs") 
    plt.xlabel("Cutoffs") 
    plt.legend()

    outputs = dict()
    outputs['cutoffs']=cutoffs
    outputs['tprs']=tprs
    outputs['fprs']=fprs
    outputs['tnrs']=tnrs
    outputs['fnrs']=fnrs
    return outputs 


def ProbClassifier(df,tags,display=False,split=True):
    # organize data relative to WT
    wtName = "wt"
    zscore = True 
    for tag in tags:
        ShapeData(df,tag,wtName,zscore=zscore,display=display)

    dTags = ["d"+t for t in tags]


    # partition in test/training set
    from sklearn.model_selection import train_test_split
    if split:
      df_train, df_test = train_test_split(df, test_size=0.3)
    else: 
      df_train = df
      df_test = df

    # compute product of conditional probabilities
    ProdCondProbs(dTags,df,df_train,display=display)

    debug = True
    if debug:
        dummy = calcRates(df_test,cutoff=None,display=display)

    # compute ROC curve for classifier
    outputs = ComputeROC(df_test,display=display)
    return outputs

