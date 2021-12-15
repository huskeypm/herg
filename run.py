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

def doit(analyses,# all, ml, prob
         bootstrap=False,display=True,
         features="all"): 
  dataFile = "feature_sets/features-latest-sets1n2.txt"
  df = pd.read_csv(dataFile, sep="\s+", comment='#')
  
  # get list of features
  if display: 
    print("Available features") 
    for col in df.columns:
      print(col)
  
  # this is the subset  I selected
  #tags = ['RMSD','WATERS','SASA','HELIX','TURNS','COILS','BETA','THREE-TEN']
  mdTags=["WATERS", "HBONDS", "RMSD", "SASA", "HELIX", "TURNS", "COILS", "THREE-TEN", "BETA"]
  bioinfTags=["CONSERVATION","FOLDX", "HYDROPHOBICITY"]
  output = ["TRAFFICKING"]
  #tags = ['RMSD','HBONDS']

  # calc addl features
  suppTags=[]
  feature.CalcInitialAA(df,newTag="INITAA")
  # uncomment to add feature 
  suppTags+=["INITAA"]
   
  feature.Calcsvolume(df,newTag="SVOLUME")
  suppTags+=["SVOLUME"]  

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

  feature.CalcAPBS(df,"feature_sets/apbs.dat",newTag="APBS")
  supptags+=["APBS"]
    
  feature.CalcSASASingleSite(df,"feature_sets/sasaSingleSite.dat",newTag="SASASingleSite")
  supptags+=["SASASingleSite"]


  # identify tags
  allTags = mdTags+bioinfTags
  if features == "mdTags":
      print("MD only") 
      tags = mdTags
  elif features =="bioinfTags":
      print("bioinf only") 
      tags = bioinfTags
  else:
      print("all features") 
      tags = allTags 
  




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
  features = "all" 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-run"):
      try: 
        analyses=sys.argv[i+1] 
      except:
        analyses="all" 

      doit(analyses,bootstrap=bootstrap,display=display,features=features)     
      quit()
    if(arg=="-nodisplay"):
      display = False
    if(arg=="-bootstrap"):
      bootstrap=True
    if(arg=="-features"):
      features=sys.argv[i+1] 
  





  raise RuntimeError("Arguments not understood")




