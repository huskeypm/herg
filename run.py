#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

#
# ROUTINE  
#

import pandas as pd
import feature 

def doit(analyses,bootstrap=False,display=True): # all, ml, prob
  dataFile = "feature_sets/features-latest-sets1n2.txt"
  df = pd.read_csv(dataFile, sep="\s+", comment='#')
  
  # get list of features
  if display: 
    print("Available features") 
    for col in df.columns:
      print(col)
  
  # this is the subset  I selected
  #tags = ['RMSD','WATERS','SASA','HELIX','TURNS','COILS','BETA','THREE-TEN']
  tags=["WATERS", "HBONDS", "RMSD", "SASA", "HELIX", "TURNS", "COILS", "THREE-TEN", "BETA"]
  output = ["TRAFFICKING"]
  #tags = ['RMSD','HBONDS']

  # calc addl features
  feature.CalcInitialAA(df,newTag="INITAA")
  # uncomment to add feature 
  tags+=["INITAA"]
  rmsf={}
  #rmsf={
  #  'RMSF36TO41' : 'rmsf_sub2.txt',
  #  'RMSF87TO92' : 'rmsf_sub3.txt',
  #  'RMSF73TO78' : 'rmsf_sub4.txt',
  #  'RMSF115TO120' : 'rmsf_sub.txt'} 
  for key,val in rmsf.items(): 
      print("feature_sets/%s"%val,key)
      #feature.CalcRMSFLoc(df,"feature_sets/rmsf_sub2.txt", newTag="RMSF36TO41")
      feature.CalcRMSFLoc(df,"feature_sets/%s"%val, newTag=key)                   
      tags+=[key]

  feature.Calcsvolume(df,newTag="SVOLUME")
  tags+=["SVOLUME"]  
    
  feature.CalcAPBS(df,"feature_sets/apbs.dat",newTag="APBS")
  tags+=["APBS"]
    
  feature.CalcSASASingleSite(df,"feature_sets/sasaSingleSite.dat",newTag="SASASingleSite")
  tags+=["SASASingleSite"]


  #display=True  # prints out indi_condprob_md.pdf; roc_conditional_probability....
  print("Computing") 
  if analyses == "prob" or analyses == "all":
    import probUtil as pU
    split = True
    outputs = pU.ProbClassifier(df,tags,display=display,split=split,bootstrap=bootstrap)  
    
  if analyses == "ml" or analyses == "all":
    import mlUtil
    outputs = mlUtil.MLClassifier(df,tags,output,display=display) 



#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -run" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  display = True
  bootstrap=False
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-run"):
      try: 
        analyses=sys.argv[i+1] 
      except:
        analyses="all" 

      doit(analyses,bootstrap=bootstrap,display=display)     
      quit()
    if(arg=="-nodisplay"):
      display = False
    if(arg=="-bootstrap"):
      bootstrap=True
  





  raise RuntimeError("Arguments not understood")




