# herg
Contains analyses for hERG classification paper

A colab notebook is at https://colab.research.google.com/drive/1g9JlkIxmejK-xgfVR7zeY9xr4BwrsjiA#scrollTo=2F8-oSXeopMa

feature_sets/ # contains feature data sets 

dataUtil.py # for conditional probabilities 
mlUtil.py # for ML strategies 


# TODOS
- something is fishy about the zscores, so I want to look into that later
- Splitting the dataset into 70% for training, 30% for testing leads to very noisy results for the test set. Most likely, in order to this properly, I will probably need to bootstrap. For now I'm using the entire data set for training and testing

# MD trajectory analysis 
- MD analysis is done with either cpptraj or tcl scripts that depends on the vmd and its associated packaged
- all MD analysis for this project are done on the local gpu cluster (faust), in view of the amount of data

- for using cpptraj:
    - we need two files: 
         1. input file: this is used to load all the trajectories and the calculation to carryout. Below is an example input file (3atp.in) used to calculate the dynamic cross correlations.

         trajin /data/jalenciks/P2X4.mg/wrap-trajectories/3atp-1.dcd 1 -1 50
         rms ../wrap-trajectories/3atp-1.pdb
         matrix correl @CA out 3atp-3mg.dat byres

            Output is written to 3atp-3mg.dat

        2. shell script: this is used to execute the input file. Below is an example.
        
        cpptraj -p 3atp.prmtop -i 3atp.in

        Note: Make sure cpptraj is installed (gpu enabled cpptraj to speed up the calculations)

- for using tcl:
    - require 2 files:
        1. tcl script with all the information necessary to load all the trajectories, paramter, topology files and the variables to calculate
        2. bash script to execute the tcl file created above

        Note: Ofcourse vmd has to be installed first to do the analysis using tcl scripts (use cuda enabled version for speed)

