#!/bin/bash

E=20 #epochs
Pe=1
Ktr=20 #Train graphs
Kva=20 #valid graphs
Kte=50 #Test graphs
dataT='clean' # load the clean data always
dataM='sim_expt' # data method
dataN='CUSTOM' # custom/ds1/ds2/ds3

# GLAD
L=15 #15 # unroll of GLAD & GRNUlar
MS='aupr' #'auc' # model select : graph recovery
GL='mse_fb' #'F_beta' #'mse' #'no'
H=3 # 'hidden layer size'
initT=1 # 'theta_init_offset'
lrG=0.05 # lr of GLAD

# GRNUlar
Hd=20 # hidden layer of DNN
lrDNN=0.05 #0.03 learning rate of NN
DNNe=200 #300 # number of epochs for fitting DNN
P=20 #Number of unrolled iterations for DNN

#GLASSO
alpha=0.01 # L1 penalty of GLASSO
mode='cv' # mode = 'cd' or 'lars' or 'cv' for graphical lasso CV

# SERGIO
NP=0.1 # Noise params
Decays=3 # Decay
SS=15 # sampling state
Sp=0.1 # sparsity

pcrln=0.2 # pcr low min
pcrlx=0.5 # pcr low max

pcrhn=0.7 # pcr high min
pcrhx=1.0 # pcr high max

kmin=1.0  # Kij_min
kmax=5.0  # Kij_max

rMR=0.1 #ratio master regulator * D
TFp=0.2 # connect prob of master tf

# GRNBOOST2, GLAD, GLASSO
tf='yes'

# Add Technical NOISE
techN='no' #'yes' # 'no' #ADD_TECHNICAL_NOISE
dropS=20 #6.5 # 6.5 or 20 dropout_shape = k
#dropP=82 #dropout_percentile = p

total_pts=1000

for D in 100 #100 #20 #100 
do
    for techN in 'no'
    do
    for dropP in 25 #75 50 #25 
    do
        for total_pts in 1000 # number of points per class
        do
            for C in 5 # number of classes
            do
            for L in 15 #30
            do
            for P in 10 # unrolled iterations for inner DNN
            do
                pts=$((${total_pts}/$C))  # number of points per class
                for Decays in 1 
                do
                    for NP in 0.1 # mean_gap
                    do
                        for NT in 'dpd' #'sp' 'spd' #'dpd'
                        do
                            for Hd in 40 #200 500 1000 #200 500
		                    do 
                            for B in 2 #5 
                                do
                            echo "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, E=$E, Pe=${Pe}, GL=${GL}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, MS=${MS}, SS=${SS}, Sp=${Sp}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, alpha=${alpha}, mode=${mode}, H=${H}, initT=${initT}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}, Hd=${Hd}, lrDNN=${lrDNN}, DNNe=${DNNe}, lrG=${lrG}, P=${P}"

#                            echo "creating data on swarm"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, SS=${SS}, Sp=${Sp}, rMR=${rMR}, TFp=${TFp}" run_sim_expt3_create_data.pbs &

#                            echo "Running GLAD on swarm-gpu"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, E=$E, Pe=${Pe}, Gl=${GL}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, MS=${MS}, SS=${SS}, Sp=${Sp}, L=${L}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, H=${H}, initT=${initT}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}, B=${B}, lrG=${lrG}" run_sim_expt3_glad.pbs &

                            echo "Running GRNUlar on gpu"
                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, E=$E, Pe=${Pe}, Gl=${GL}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, MS=${MS}, SS=${SS}, Sp=${Sp}, L=${L}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, H=${H}, initT=${initT}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}, Hd=${Hd}, lrDNN=${lrDNN}, DNNe=${DNNe}, lrG=${lrG}, P=${P}, B=${B}" run_expt_grnular.pbs &

#                            echo "Running GRNUlar on hive-gpu"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, E=$E, Pe=${Pe}, Gl=${GL}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, MS=${MS}, SS=${SS}, Sp=${Sp}, L=${L}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, H=${H}, initT=${initT}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}, Hd=${Hd}, lrDNN=${lrDNN}, DNNe=${DNNe}, lrG=${lrG}, P=${P}" run_sim_expt3_grnular_hive.pbs &

#                            echo "Running GRNUlar on swarm"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, E=$E, Pe=${Pe}, Gl=${GL}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, MS=${MS}, SS=${SS}, Sp=${Sp}, L=${L}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, H=${H}, initT=${initT}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}, Hd=${Hd}, lrDNN=${lrDNN}, DNNe=${DNNe}, lrG=${lrG}, P=${P}" run_sim_expt3_grnular_swarm.pbs &

#                            echo "Running GLAD on swarm"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, E=$E, Pe=${Pe}, Gl=${GL}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, MS=${MS}, SS=${SS}, Sp=${Sp}, L=${L}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, H=${H}, initT=${initT}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}" run_sim_expt3_glad_swarm.pbs &

#                            echo "Running GLASSO on swarm"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, SS=${SS}, Sp=${Sp}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, alpha=${alpha}, mode=${mode}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}" run_sim_expt3_glasso.pbs &

#                            echo "Running GRNBoost2 on swarm"
#                            qsub -v "D=$D, pts=${pts}, C=$C, NP=${NP}, Decays=${Decays}, dataT=${dataT}, NT=${NT}, Ktr=${Ktr}, Kva=${Kva}, Kte=${Kte}, SS=${SS}, Sp=${Sp}, tf=${tf}, pcrln=${pcrln}, pcrlx=${pcrlx}, pcrhn=${pcrhn}, pcrhx=${pcrhx}, kmin=${kmin}, kmax=${kmax}, tf=${tf}, rMR=${rMR}, TFp=${TFp}, techN=${techN}, dropS=${dropS}, dropP=${dropP}, dataM=${dataM}, dataN=${dataN}" run_sim_expt3_grnboost2.pbs &
                            done
                            done
                        done
                    done
                done
            done
            done
            done
        done
    done
    done
done
wait
echo "All scripts submitted"
