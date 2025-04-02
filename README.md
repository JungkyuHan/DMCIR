# DMCIR

This code is the implementation of the paper "Conditional Diffusion Model as a Base for Cold Item Recommendation".

Tested in Tensorflow 2.15 and python 3.9

Required Packages: numpy, tqdm, pandas, scipy

Run : move to src diretory, and 
  - CiteULike :
    -- Diffusion based generator : python _01_01_DMCIR_main__tr_CiteULike.py                                       
    -- Refiner                   : python _01_02_DMCIRPlus_main_CiteULike.py
  - ML25M 
    -- Diffusion based generator : python _02_01_DMCIR_main__tr_ML25M.py                                       
    -- Refiner                   : python _02_02_DMCIRPlus_main_ML25M.py

