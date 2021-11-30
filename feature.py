
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

"""
Brief Documentation of feature 
"""
def CalcRMSFScore(rmsfFileName, variantName, aaRange=[]):
    return 1/0

"""
Brief Documentation of feature 
"""
def CalcNativeScore(nativeFileName, variantName):
    return 1/0

"""
Brief Documentation of feature 
"""
def CalcSASASingleScore(nativeFileName, variantName):
    return 1/0

