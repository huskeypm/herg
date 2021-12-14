
"""
This feature assigns a tag to the initial identity of an a.a.
e.g. A-->X would return the value 0, based on the properties table below
THe new data is assigned to a new dataframe tag (INITAA)
"""
def CalcInitialAA(df,newTag="INITAA"):

  properties={
      'a': 0, 'c': 1, 'd': 2, 'e': 3, 'f': 4,
      'g': 5, 'h': 6, 'i': 7, 'k': 8, 'l': 9,
      'm':10, 'n':11, 'p':12, 'q':13, 'r':14,
      's':15, 't':16, 'v':17, 'w':18, 'y':19,
  }

  def calc_prop(variantName):
    #print(variantName)
    if variantName == 'wt':
      return 20

    daID = properties[variantName[0]]
    return daID 
   
  df[newTag] = df.apply( lambda row:
         calc_prop(row['VARIANT']),
         axis=1)


##### FOR KAL TO IMPLEMENT ###############
import pandas as pd

"""
Brief Documentation of feature 
"""
def CalcRMSFLoc(df,rmsfFileName,newTag="RMSF" ):#, aaRange=[]):
    # convert into dictionary for easy lookup  
    #print( df.loc[df['VARIANT'] == 'a57p'] ) 
    df.loc[:,newTag] = 0      
    with open(rmsfFileName) as f: 
      for line in f:
        # his format has a header with a single entry
        vals = line.split()
        if len(vals)<2:
            continue 

        mut,val = vals 
        idx = df.index 
        cond = df['VARIANT']==mut      #vals[0] ) 
        varIdx = idx[cond].tolist()

        if len(varIdx)<1:
            print(mut, " not found in input list")
            continue 

        df.loc[varIdx,newTag] = float( val ) 
        



"""
Brief Documentation of feature 
"""
def CalcNativeScore(nativeFileName, variantName):
    return 1/0

"""
Brief Documentation of feature.
data is in file 'feature_sets/sasa-wt-aa.txt'
"""
def CalcSASASingleScore(nativeFileName, variantName):
    return 1/0



####### change im aa sidechain volume is estimated.
####### volume in solution is being used for this calculation
####### volumes are missig for his, arg, cys
####### ref: "Volumes of Individual Amino Acid Residues in Gas-Phase Peptide Ions"
def Calcsvolume(df,newTag="SVOLUME"):
    properties={
      'a': 100.3, 'c':100, 'd': 113.1, 'e': 140.2, 
      'f': 202.3, 'g': 71.7, 'h':202, 'i': 175.4,
      'k': 170.3, 'l': 178.7, 'm':174.9, 'n':128.4,
      'p':137.2, 'q':156, 'r':170, 's':100.7, 't':127.6,
      'v':150.6, 'w':239, 'y':205.3,
    }

    def calc_prop(variantName):
        #print(variantName,variantName[0],variantName[-1])
        if variantName == 'wt':
          return 0
        
        # compute volume difference between wt and mutated residues
        volWT = properties[variantName[0]]
        volMut = properties[variantName[-1]]
        volDiff = abs(volWT-volMut) 
        return volDiff 

    df[newTag] = df.apply( lambda row:
         calc_prop(row['VARIANT']),
         axis=1)


####### aa charges
def Calccharge(df,newTag="CHARGE"):
  properties={
      'a':0, 'c':0, 'd':-1, 'e':-1, 
      'f':0, 'g':0, 'h':1, 'i':0,
      'k':1, 'l':0, 'm':0, 'n':0,
      'p':0, 'q':0, 'r':1, 's':0, 't':0,
      'v':0, 'w':0, 'y':0,
  }
