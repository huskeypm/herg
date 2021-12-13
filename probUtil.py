import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#pd.options.mode.chained_assignment = 'raise'


def CalcZScore(df,tag):
    vals = df.loc[df.index,[tag]]
    avg = np.mean(vals)
    stddev = np.std(vals)
    zs = (vals - avg)/stddev
    return zs

# compute relative values
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
        df.loc[:,newTag] = df[tag] - vWT
        label = newTag
    else:
        df.loc[:,newTag] = vWT - df[tag]    # lower values have higher pctg LOF variants
        label = "-1 x "+newTag
    #print(tag,pctSub,pctSup,tot,label)

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


#    plt.plot(threshVals,probs)
    if display:
        # save data
        stacked = np.stack([threshVals,probs], axis=0)
        np.savetxt(fileName,stacked) 
        # plot 
        plt.plot(threshVals,probs,label=tag)

        plt.ylabel('Prob(Insoluble|Threshold)')
        plt.xlabel('Threshold')
        plt.legend()
        plt.title("Cond. probability") 

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
def ProdCondProbs(tagsSubset,df,df_train,df_test,display=False):
    if display: 
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
        df.loc[:,'pInSol'+tag] = ps
        #print(vals)
        #print(ps)
        prods *= ps

    #df['Prod'] = prods

    #totProd = np.sum(prods)
    maxProd = np.max(prods)
    #print(totProd)
    df.loc[:,'Prod'] = prods/maxProd
    # verify that np.sum(df['Prod']) = 1
    
    # apply to test/train data 
    df_train.loc[:,"Prod"] = 0
    df_test.loc[:,"Prod"] = 0
    #df_train = df_train.assign( Prod=0 )   
    #df_test  = df_test.assign( Prod=0 )   
    #df_test.loc[:,"Prod"] = 0
    for index, row in df.iterrows():
      #print(row['VARIANT'])
      refVar = row['VARIANT']
      idxTrain = df_train.index[df_train['VARIANT'] == row['VARIANT']]
      idxTest = df_test.index[df_test['VARIANT'] == row['VARIANT']]

      if len(idxTest)>0: # shouldn't get more than 1, but this should throw an error if so
        df_test.loc[idxTest[0],'Prod']=row['Prod']
      if len(idxTrain)>0: # shouldn't get more than 1, but this should throw an error if so
        df_train.loc[idxTrain[0],'Prod']=row['Prod']

    #print(df[['VARIANT','Prod']])
    #print(df_train[['VARIANT','Prod']])
    #print(df_test[['VARIANT','Prod']])
    


    if display: 
      plt.savefig('indi_condprob_md.pdf')

      #plt.figure()
      #plt.plot(df['Prod']) 
      #plt.savefig("prod.png") 




# The product of conditional probabilities gives the likelihood that a member is defective (insoluble), given the products of being insoluble for each feature. We then pick a 'reasonable' threshold to decide if a member is insoluble

def calcRates(df,cutoff=None,display=False,verbose=False):
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
        plt.gcf().savefig("prod.png") 

    insolSupThresh = np.where(insolVs >= cutoff)
    nInsolSupThresh = np.shape(insolSupThresh)[1]
    insolSubThresh = np.where(insolVs < cutoff)
    nInsolSubThresh = np.shape(insolSubThresh)[1]

    solSupThresh = np.where(solVs >= cutoff)
    nsolSupThresh = np.shape(solSupThresh)[1]
    solSubThresh = np.where(solVs< cutoff)
    nsolSubThresh = np.shape(solSubThresh)[1]

    TP = nInsolSupThresh
    FP = nsolSupThresh
    TN = nsolSubThresh
    FN = nInsolSubThresh 

    # TP/(TP + FP)
    positives = np.float( TP + FP )
    if positives <= 1e-3:
      TPR=0.
      FPR=0.
    else: 
      TPR = TP/positives                                      
      # FP/(TP + FP)
      FPR = FP/positives

    negatives = np.float( FN + TN )
    if negatives <= 1e-3:
        FNR = 0.
        TNR = 0.
    else:
        # FN/(FN + TN) 
        FNR = FN/negatives 
        # TN/(FN + TN) 
        TNR = TN/negatives 

    if verbose:
        print("Cutoff ",cutoff)
        print("TPR (insol)", TPR, " Support ", TP)
        print("FPR (insol)", FPR, " Support ", FP)
        print("TNR (insol)", TNR, " Support ", TN)
        print("FNR (insol)", FNR, " Support ", FN)


    outputs = {
            "TP":TP,
            "FP":FP,
            "TN":TN,
            "FN":FN,
            "TPR":TPR,
            "FPR":FPR,
            "TNR":TNR,
            "FNR":FNR
    } 
    return outputs

# calc F1 score, etc
def calcStats(df_test,cutoff=0.3,display=False,verbose=False):
  # stats for nontrafficking 
  output = calcRates(df_test,cutoff=cutoff,display=display)
  TP,FP,TN,FN = output['TP'],output['FP'],output['TN'],output['FN']

  def calcall(TP,FP,TN,FN):
    accuracy = (TP+TN)/np.float(TP+TN+FP+FN) 
    try:
      precision = TP/(TP + FP)
    except:
      precision = 0.

    try: 
        recall = TP/(TP + FN) 
    except:
        recall=0.


    try:
        F1 = 2/(1/recall + 1/precision) 
    except:
        F1 = 0.
    return accuracy,precision,recall,F1 

  accuracy,precision,recall,F1 = calcall(TP,FP,TN,FN)
  if verbose: 
    print("Accuracy(NT)", accuracy) 
    print("Precision(NT)", precision) 
    print("Recall(NT)", recall)             
    print("F1 score(NT)", F1) 

  # get for trafficking now (a true negative for NT is a true pos. for T (a true negative for NT is a true pos. for T)) 
  tTN, tFN, tTP, tFP = TP,FP,TN,FN
  taccuracy,tprecision,trecall,tF1 = calcall(tTP,tFP,tTN,tFN)
  if verbose: 
    print("Accuracy(T)", taccuracy) 
    print("Precision(T)", tprecision) 
    print("Recall(T)", trecall)             
    print("F1 score(T)", tF1) 

  statsOut = {
          'accT':taccuracy ,
          'precT':tprecision,
          'recT':trecall,
          'f1T':tF1,        

          'acc':accuracy ,
          'prec':precision,
          'rec':recall,
          'f1':F1,        
          }
  return statsOut


# compute ROC curve
def ComputeROC(df, threshVals = 20,display=False):
    prods = df['Prod']
    minThresh = np.min(prods)
    maxThresh = np.max(prods)
    cutoffs = np.linspace(minThresh,maxThresh, threshVals)

    tprs = []
    fprs = []
    tnrs = []
    fnrs = []
    for cutoff in cutoffs:
        output = calcRates(df,cutoff=cutoff,display=False,verbose=False)
        tpr,fpr,tnr,fnr = output['TPR'],output['FPR'],output['TNR'],output['FNR']
        tprs.append(tpr)
        fprs.append(fpr)
        tnrs.append(tnr)
        fnrs.append(fnr)

    # add 1,1 point
    tprs.append(1)
    fprs.append(1)
    fnrs.append(1)
    tnrs.append(1)
    tprs = np.asarray(tprs)
    fprs = np.asarray(fprs)
    fnrs = np.asarray(fnrs)
    tnrs = np.asarray(tnrs)

    from sklearn.metrics import auc
    rfnrs = fnrs# + 1e-2*np.arange( len(fnrs ))
    idx = np.argsort(rfnrs)
    rfnrs = np.sort(rfnrs)
    rtprs = tprs[idx]
    daAUC = auc(rfnrs,rtprs)

     
    if display: 
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
      plt.gcf().savefig("rates.png") 

    outputs = dict()
    outputs['cutoffs']=cutoffs
    outputs['tprs']=tprs
    outputs['fprs']=fprs
    outputs['tnrs']=tnrs
    outputs['fnrs']=fnrs
    outputs['auc']=auc  
    return outputs 


def BuildAndEvaluate(df,dTags,cutoff=0.04,display=False,split=True):
    if split:
      df_train, df_test = train_test_split(df, test_size=0.3)
      #print ( len(df),len(df_train), len(df_test) )
    else: 
      df_train = df
      df_test = df

    # compute product of conditional probabilities
    ProdCondProbs(dTags,df,df_train,df_test,display=display)

    cutoff = 0.04     # user encoded 
    #print("TRAINING stats") 
    outs_train= calcStats(df_train,cutoff=cutoff,display=display)
    #print("TESTING  stats") 
    outs_test= calcStats(df_test,cutoff=cutoff,display=display)

    return outs_train,outs_test, df_test


def ProbClassifier(df,tags,display=False,split=True,bootstrap=False):
    # organize data relative to WT
    wtName = "wt"
    zscore = True 
    for tag in tags:
        ShapeData(df,tag,wtName,zscore=zscore,display=display)
    dTags = ["d"+t for t in tags]


    # partition in test/training set
    if bootstrap:
      nIter = 500
      print("Bootstrap with ",nIter)
      statsdf = pd.DataFrame()
      for i in range(nIter):
        print(i)
        outs_train, outs_test,df_test = BuildAndEvaluate(
                df,dTags,cutoff=0.04,split=split,display=False)     
        dfn = pd.DataFrame(outs_test,index=[i])
        dfn['AccTraining'] = outs_train['acc']
        statsdf = statsdf.append(dfn)

      print(statsdf.head())
      for column in statsdf.columns[:]:
          print(column+" avg: ",statsdf[column].mean())

    else:
        outs_train, outs_test,df_test = BuildAndEvaluate(
                df,dTags,cutoff=0.04,split=split,display=display)


    # compute ROC curve for classifier
    outputs = ComputeROC(df_test,display=display)

    return outputs

