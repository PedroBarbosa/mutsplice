#!/bin/bash
#SBATCH --job-name=lsgkm_clf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-71%1
#SBATCH --output=lsgkm_clf_%j.log

data=$(readlink -f "$1")

readarray rbps < $(readlink -f $data)
rbp=$(echo "${rbps[$SLURM_ARRAY_TASK_ID]}" | tr -d '\n')
pos="${rbp}/clf/positive.fa"
neg="${rbp}/clf/negative.fa"
out="${rbp}/clf/out"

# -y 0 -> SVM type: C_SVC
# -l 9 -> full length considered with gaps
# -k 6 -> non-gapped positions
# -d 3 -> max number of mismatches
# -m 10000 -> Cache memory size (10Gb)
# -x 5 -> N-folder cross validation
# -T 16 -> Number of threads for parallel calculation

# -t 2 -> Kernel function: estimated l-mer with truncated filter (gkm)
linear_kernel="/home/pbarbosa/git_repos/lsgkm-svr/bin/gkmtrain -t2 -y0 -l9 -k6 -d3 -x5 -m10000 -T16 ${pos} ${neg} ${out}_gkm"

# -g 2 -> Gamma for RBF Kernel
# -c 10 -> Regularization parameter for SVM-C
# -t 3 -> Kernel function: gkm + RBF (gkmrbf)
non_linear_kernel="/home/pbarbosa/git_repos/lsgkm-svr/bin/gkmtrain -t3 -y0 -l9 -k6 -d3 -c10 -x5 -m10000 -T16 -g2 -c10 ${pos} ${neg} ${out}_gkmrbf"

srun $linear_kernel
srun $non_linear_kernel
