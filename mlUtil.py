# -*- coding: utf-8 -*-
""" Install these if needed" 
#!pip3 install pydotplus
#!pip3 install mlxtend

"""### Load requisite libraries"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import itertools
import matplotlib.gridspec as gridspec

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, plot_roc_curve

from mlxtend.data import iris_data
from sklearn.utils import resample

import os
import math
from math import floor, ceil

"""Stuff for plotting trees"""

from six import StringIO
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

"""plot data"""

def plotDT(dtree,featureNames = None,classNames=None):
    dot_data = StringIO()
    export_graphviz(dtree, 
                    out_file=dot_data,  
                    feature_names = featureNames,
                    class_names =  classNames,
                    filled=True, 
                    rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    l=Image(graph.create_png())
    return l

def TestAccuracy(clf,X_test,y_test, display=False):
    
    y_pred  = clf.predict( X_test.values)  
    results = ( y_test.values[:,0]==y_pred )

    if display:
        nEntries, dummy = X_test.shape
        for i in range(nEntries):
            print(X_test.iloc[i])
            print("Class=%s Prediction %s %s\n"%(y_test.values[i,0],y_pred[i],results[i]) )
        #print("prediction",results)

    # accuracy
    nTrue = np.sum( results )
    nTot = np.float( np.shape( results)[0] )
    accuracy = nTrue/nTot
    print("Overall accuracy ",accuracy)

def TestAccuracySingle(clf,X_test,Y_test,idx=0):
    
    print(X_test.iloc[idx])
    result = clf.predict( [X_test.values[idx,:]])
    print("prediction",result)
    #print(Y_test.iloc[idx])

#TestAccuracy(clf_entropy,X_test,Y_test,display=True)

def DataCuration(df):
    #df['decimal_place_2'] = df['decimal_place_2'].round(2)
    df['HELIX'] = df['HELIX'].round(2)
    df['TURNS'] = df['TURNS'].round(2)
    df['COILS'] = df['COILS'].round(2)
    df['THREE-TEN'] = df['THREE-TEN'].round(2)
    df['BETA'] = df['BETA'].round(2)
    df['HBONDS'] = df['HBONDS'].round()
    df['WATERS'] = df['WATERS'].round()
    df['RMSD'] = df['RMSD'].round(2)
    df['SASA'] = df['SASA'].round()



"""### Example with PAS data """

import pandas as pd

def run():
    
    dataFile = "feature_sets/features-latest-sets1n2.txt"
    df = pd.read_csv(dataFile, sep="\s+", comment='#')

    features=["WATERS", "HBONDS", "RMSD", "SASA", "HELIX", "TURNS", "COILS", "THREE-TEN", "BETA"]
    output = ["TRAFFICKING"]

    MLClassifier(df, features, output)

"""
Initializes, trains and tests ML classifiers
Written for decision tree currently 
"""
def MLClassifier(df,features,output, display=False,random_state=50):
    # rounding for some reason 
    DataCuration(df)


    # get featrures/output 
    X = df[features]
    Y = df[output]

    
    # training 
    X_train, X_test, Y_train, Y_test =\
      train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    
    
    #initializing the classifier and fitting it
    clf_entropy = DecisionTreeClassifier(
      criterion = "entropy",
      #class_weight="balanced",
      random_state = random_state, #set random number seed
        max_depth=None,
        max_features=3,
        min_samples_leaf=1) #min of samples needed at a node for it to split further
    
    print("Add me")
    ##>>> add clf_rf = 
    
    # fitting 
    model=clf_entropy.fit(X_train, Y_train)

    #classNames=np.array(['0', '1']) # need to verify, but I think this is correct
    #classNames=np.array(['benign', 'pathogenic']) # need to verify, but I think this is correct
    classNames=np.array(['non-trafficking', 'trafficking']) # need to verify, but I think this is correct
    
    if display: 
      plotDT(clf_entropy,featureNames=features, classNames=classNames)
      plt.savefig('dt_tree_md.pdf')

    
    # print training accuracy
    print("Training sample accuracy")
    TestAccuracy(clf_entropy,X_train,Y_train)
    
    #TestAccuracy(clf_entropy,X_test,Y_test, display=True)
    print('Test Accuracy')
    TestAccuracy(clf_entropy,X_test,Y_test)
    
    
    ### Print overall classification metrics
    report = True
    if report: 
      y_predict = clf_entropy.predict(X_test.values)
        
      print("extract class report data")
      print(classification_report(Y_test, y_predict, target_names=classNames))
      #print(confusion_matrix(Y_test, y_predict))
    
    if display:
        plt.figure()  
        plot_confusion_matrix(clf_entropy, X_test, Y_test)
        plt.savefig('dt_cm_md.pdf')
        
        plt.figure()
        plot_roc_curve(clf_entropy, X_test, Y_test)
        plt.savefig('dt_roc_mdfeatures_latest.pdf')
        
        # print feature importance
        plt.figure()
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh', color="green")
        plt.title('Feature Importances-MD features (DT)')
        plt.savefig('dt_fi_md.pdf')

def OLD():
    """### generate data for ROC curve"""
    
    HEX = df[['HELIX']]
    B = df[["BETA"]]
    T = df[["TURNS"]]
    C = df[["COILS"]]
    TT = df[["THREE-TEN"]]
    R = df[["RMSD"]]
    W = df[["WATERS"]]
    H = df[["HBONDS"]]
    S = df[["SASA"]]
    
    Y = df["TRAFFICKING"]
    #X = df[features]
    
    #extracting metrics for whole model
    y_predict = clf_entropy.fit(X_train, Y_train).predict_proba(X_test)
    #y_predict = clf_entropy.fit(X_train, Y_train).decisions(X_test)
    fpr, tpr, thr = metrics.roc_curve(Y_test, y_predict[:,1], drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)
    
    
    #redifined the classifier for one feature
    clf2_dt = DecisionTreeClassifier(criterion = "entropy",random_state=random_state,max_depth=None, min_samples_leaf=1)
    
    #redifined the classifier for one feature
    clf2_dt = DecisionTreeClassifier(criterion = "entropy",random_state=random_state,max_depth=None, min_samples_leaf=1)
    
    #helicity
    #split
    HEX_train, HEX_test, HEY_train, HEY_test = train_test_split(HEX, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #train
    #modelHE=clf2_dt.fit(HEX_train, HEY_train.values.ravel())
    #train
    #HEY_predict = clf2_dt.predict(HEX_test.values)
    HEY_predict = clf2_dt.fit(HEX_train, HEY_train.values.ravel()).predict_proba(HEX_test.values)
    f_helix, t_helix, th_helix = metrics.roc_curve(HEY_test, HEY_predict[:,1], pos_label=1, drop_intermediate=False)
    helix_auc = metrics.auc(f_helix, t_helix)
    
    #beta
    bx_train, bx_test, by_train, by_test = train_test_split(B, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelB=clf2_dt.fit(bx_train, by_train.values.ravel())
    #by_predict = clf2_dt.predict(bx_test.values)
    by_predict = clf2_dt.fit(bx_train, by_train).predict_proba(bx_test)
    f_beta, t_beta, th_beta = metrics.roc_curve(by_test, by_predict[:,1], pos_label=1, drop_intermediate=False)
    beta_auc = metrics.auc(f_beta, t_beta)
    
    #coil
    cx_train, cx_test, cy_train, cy_test = train_test_split(C, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelC=clf2_dt.fit(cx_train, cy_train.values.ravel())
    #cy_predict = clf2_dt.predict(cx_test.values)
    cy_predict = clf2_dt.fit(cx_train, cy_train).predict_proba(cx_test)
    f_coil, t_coil, th_coil = metrics.roc_curve(cy_test, cy_predict[:,1], pos_label=1, drop_intermediate=False)
    coil_auc = metrics.auc(f_coil, t_coil)
    
    #3-10
    ttx_train, ttx_test, tty_train, tty_test = train_test_split(TT, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelTT=clf2_dt.fit(ttx_train, tty_train.values.ravel())
    #tty_predict = clf2_dt.predict(ttx_test.values)
    tty_predict = clf2_dt.fit(ttx_train, tty_train).predict_proba(ttx_test)
    f_tten, t_tten, th_tten = metrics.roc_curve(tty_test, tty_predict[:,1], pos_label=1, drop_intermediate=False)
    tten_auc = metrics.auc(f_tten, t_tten)
    
    #rmsd
    #extract metrics for rmsd. f_rmsd = false positive rate, t_rmsd=true positive rate, th_rmsd=threshold values
    rx_train, rx_test, ry_train, ry_test = train_test_split(R, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelR=clf2_dt.fit(rx_train, ry_train.values.ravel())
    #ry_predict = clf2_dt.predict(rx_test.values)
    ry_predict = clf2_dt.fit(rx_train, ry_train).predict_proba(rx_test)
    f_rmsd, t_rmsd, th_rmsd = metrics.roc_curve(ry_test, ry_predict[:,1], pos_label=1, drop_intermediate=False)
    rmsd_auc = metrics.auc(f_rmsd, t_rmsd)
    
    #extract metrics for sasa
    sx_train, sx_test, sy_train, sy_test = train_test_split(S, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelS=clf2_dt.fit(sx_train, sy_train.values.ravel())
    #sy_predict = clf2_dt.predict(sx_test.values)
    sy_predict = clf2_dt.fit(sx_train, sy_train).predict_proba(sx_test)
    f_sasa, t_sasa, th_sasa = metrics.roc_curve(sy_test, sy_predict[:,1], pos_label=1, drop_intermediate=False)
    sasa_auc = metrics.auc(f_sasa, t_sasa)
    
    #hbonds
    #extract metrics for hbonds
    hx_train, hx_test, hy_train, hy_test = train_test_split(H, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelH=clf2_dt.fit(hx_train, hy_train.values.ravel())
    #hy_predict = clf2_dt.predict(hx_test.values)
    hy_predict = clf2_dt.fit(hx_train, hy_train).predict_proba(hx_test)
    f_hbonds, t_hbonds, th_hbonds = metrics.roc_curve(hy_test, hy_predict[:,1], pos_label=1, drop_intermediate=False)
    hbonds_auc = metrics.auc(f_hbonds, t_hbonds)
    
    #water
    #extract metrics for water
    wx_train, wx_test, wy_train, wy_test = train_test_split(W, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    #modelW=clf2_dt.fit(wx_train, wy_train.values.ravel())
    #wy_predict = clf2_dt.predict(wx_test.values)
    wy_predict = clf2_dt.fit(wx_train, wy_train).predict_proba(wx_test)
    f_water, t_water, th_water = metrics.roc_curve(wy_test, wy_predict[:,1], pos_label=1)
    water_auc = metrics.auc(f_water, t_water)
    
    #turns
    tx_train, tx_test, ty_train, ty_test = train_test_split(T, Y, train_size=0.7, test_size=0.3, random_state=random_state)
    modelT=clf2_dt.fit(tx_train, ty_train.values.ravel())
    #ty_predict = clf2_dt.predict(tx_test.values)
    ty_predict = clf2_dt.fit(tx_train, ty_train).predict_proba(tx_test)
    f_turns, t_turns, th_turns = metrics.roc_curve(ty_test, ty_predict[:,1], pos_label=1, drop_intermediate=False)
    turns_auc = metrics.auc(f_turns, t_turns)
    
    """### Plot ROC"""
    
    #plot the roc curves of rmsd, waters and hbonds. auc is area under the curve
    #plot_roc_curve(clf_entropy, X_test, Y_test)
    plt.figure()
    plt.plot (fpr, tpr, color='cyan', label="DT Classifier(AUC=%0.2f)"% auc)
    if False: 
      plt.plot (f_tten, t_tten, label="3-10 (AUC=%0.2f)"% tten_auc)
      plt.plot(f_helix, t_helix, label="HELIX(AUC=%0.2f)"% helix_auc)
      plt.plot (f_beta, t_beta, label="BETA (AUC=%0.2f)"% beta_auc)
      plt.plot (f_sasa, t_sasa, label="SASA (AUC=%0.2f)"% sasa_auc)
      plt.plot (f_water, t_water, label="Waters (AUC=%0.2f)"% water_auc)
      plt.plot (f_coil, t_coil, label="Coils (AUC=%0.2f)"% coil_auc)
      plt.plot (f_turns, t_turns, label="TURNS (AUC=%0.2f)"% turns_auc)
      plt.plot(f_rmsd, t_rmsd, label="RMSD(AUC=%0.2f)"% rmsd_auc)
      plt.plot (f_hbonds, t_hbonds, label="H-Bonds (AUC=%0.2f)"% hbonds_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random')
    
    plt.title('ROC curve-MD features (DT)')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    #plt.legend(loc="upper left")
    
    plt.legend(bbox_to_anchor=(1,0.9), loc='upper left')
    plt.savefig('dt_roc_md.pdf', bbox_inches='tight')

print("Select F1score for non/trafficking ") 


run()
