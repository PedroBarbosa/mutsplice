#!/bin/bash
#SBATCH --job-name=lsgkm_rgr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-71%1
#SBATCH --output=lsgkm_rgr_%j.log

data=$(readlink -f "$1")

readarray rbps < $(readlink -f $data)
rbp=$(echo "${rbps[$SLURM_ARRAY_TASK_ID]}" | tr -d '\n')
seqs="${rbp}/regr/seqs.fa"
preds="${rbp}/regr/spliceai_pred.txt"
out="${rbp}/regr/out"

# -y 3 -> SVM type: EPSILON_SVR (regression)
# -l 9 -> full length considered with gaps
# -k 6 -> non-gapped positions
# -d 3 -> max number of mismatches
# -m 10000 -> Cache memory size (10Gb)
# -x 5 -> N-folder cross validation
# -T 16 -> Number of threads for parallel calculation
# -p 0.1 regression epsilon to tolerate magin of error
# -t 2 -> Kernel function: estimated l-mer with truncated filter (gkm)
linear_kernel="/home/pbarbosa/git_repos/lsgkm-svr/bin/gkmtrain -t2 -y3 -l9 -k6 -d3 -x5 -m10000 -T16 -p 0.1 ${seqs} ${preds} ${out}_gkm"

# -g 2 -> Gamma for RBF Kernel
# -c 10 -> Regularization parameter for SVM-C
# -t 3 -> Kernel function: gkm + RBF (gkmrbf)
non_linear_kernel="/home/pbarbosa/git_repos/lsgkm-svr/bin/gkmtrain -t3 -y3 -l9 -k6 -d3 -c10 -x5 -m10000 -T16 -g2 -c10 -p 0.1 ${seqs} ${preds} ${out}_gkmrbf"

srun $linear_kernel
srun $non_linear_kernel
