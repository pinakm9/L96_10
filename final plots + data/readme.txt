0 -> observation covariance 1.0
1 -> observation covariance 0.1 

pc -> particle count

config_1 -> bias 0.0, covariance 0.1
config_2 -> bias 2.0, covariance 0.5
config_3 -> bias 4.0, covariance 1.0 


############################
Folder structures for L96_10
############################
__________________________
plots (contains png plots)
__________________________
|
|____0
|    |
|    |____dist_1_vs_2.png
|    |____dist_1_vs_3.png
|    |____dist_2_vs_3.png
|
|____1
|    |
|    |____dist_1_vs_2.png
|    |____dist_1_vs_3.png
|    |____dist_2_vs_3.png
|
|____enkf
     |
     |____enkf_200_vs_bpf_250.png
     |____enkf_200_vs_bpf_1000.png
     |____enkf_200_vs_bpf_2000.png
     |____bpf_250_vs_bpf_2000.png
      

________________________________________________________________________________________________________________
dists (contains sqrt of Sinkhorn divergence between particle filters with same number of particles as csv files)
________________________________________________________________________________________________________________
|
|____0_pc_250
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_500
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_1000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_2000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_250
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_500
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_1000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_2000
     |____1_vs_2.csv
     |____1_vs_3.csv
     |____2_vs_3.csv

________________________________________
initial_dists (initial dists for insets)
________________________________________
|
|____0_pc_250
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_500
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_1000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_2000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_250
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_500
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_1000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_2000
     |____1_vs_2.csv
     |____1_vs_3.csv
     |____2_vs_3.csv
_____________________________________________________________________
cov (contains largest eigenvalue of analysis covariance as csv files)
_____________________________________________________________________
|
|____0_pc_250
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____0_pc_500
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____0_pc_1000
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____0_pc_2000
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____1_pc_250
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____1_pc_500
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____1_pc_1000
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____1_pc_2000
     |____config_1.csv
     |____config_2.csv
     |____config_3.csv 

______________________________________________________________________________________________________________
enkf_dists (contains sqrt of Sinkhorn divergence between enkf(200 memebers) and particle filters as csv files)
______________________________________________________________________________________________________________
|
|____0_pc_250
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____0_pc_1000
|    |____config_1.csv
|    |____config_2.csv
|    |____config_3.csv
|
|____0_pc_2000
     |____config_1.csv
     |____config_2.csv
     |____config_3.csv

____________________________________________________________________________________________________________________________
bpf_dists (contains sqrt of Sinkhorn divergence between particle filters with different number of particles for same config)
____________________________________________________________________________________________________________________________
|
|____0
     |____config_1_250_vs_2000.csv
     |____config_2_250_vs_2000.csv
     |____config_3_250_vs_2000.csv

##########################
Folder structures for L63
##########################
__________________________
plots (contains png plots)
__________________________
|
|____0
|    |
|    |____dist_1_vs_2.png
|    |____dist_1_vs_3.png
|    |____dist_2_vs_3.png
|
|____1
     |
     |____dist_1_vs_2.png
     |____dist_1_vs_3.png
     |____dist_2_vs_3.png

________________________________________________________________________________________________________________
dists (contains sqrt of Sinkhorn divergence between particle filters with same number of particles as csv files)
________________________________________________________________________________________________________________
|
|____0_pc_100
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_200
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_400
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_800
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_100
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_200
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_400
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_800
     |____1_vs_2.csv
     |____1_vs_3.csv
     |____2_vs_3.csv


________________________________________
initial_dists (initial dists for insets)
________________________________________
|
|____0_pc_250
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_500
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_1000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____0_pc_2000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_250
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_500
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_1000
|    |____1_vs_2.csv
|    |____1_vs_3.csv
|    |____2_vs_3.csv
|
|____1_pc_2000
     |____1_vs_2.csv
     |____1_vs_3.csv
     |____2_vs_3.csv