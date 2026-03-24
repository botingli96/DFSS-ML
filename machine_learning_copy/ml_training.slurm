#!/bin/bash
#SBATCH --job-name=FCNN_Bz
#SBATCH --mem-per-cpu=16G

#SBATCH --output=%x_%j.out            # Standard output file (%x = job-name, %j = job ID)
#SBATCH --error=%x_%j.err             # Standard error file (%x = job-name, %j = job ID)

#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=botingl@princeton.edu
#SBATCH -A pppl

module purge
export PATH=/home/botingl/miniconda3/bin:$PATH
unset PYTHONPATH
unset PYTHONHOME

# Run the script using conda run
# conda run -n portal_new python /home/botingl/machine\ learning\ copy/SBI_Bz.py
# conda run -n portal_new python /home/botingl/machine\ learning\ copy/SBI_Bz_correct.py
# conda run -n portal_new python /home/botingl/machine\ learning\ copy/SBI_Bz_diff_sets.py
# conda run -n portal_new python /home/botingl/machine\ learning\ copy/SBI_Bz_diff_sets_evaluation.py
# conda run -n portal_new python /home/botingl/machine\ learning\ copy/SBI_Bz_diff_sets_evaluation_testonly.py
# conda run -n portal_new python /home/botingl/machine\ learning\ copy/SBI_Bz_UQ.py

# conda run -n tensor python /home/botingl/machine\ learning\ copy/CNN_Bz.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/CNN_paper_Bz.py

# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz_correct.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz_noprep.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz_diff_sets.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz_diff_sets_evaluation.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz_diff_sets_evaluation_noprep.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN_Bz_UQ.py

# conda run -n tensor python /home/botingl/machine\ learning\ copy/GPR_outliar.py

# conda run -n tensor python -u /home/botingl/machine\ learning\ copy/GBM_XGB.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_XGB_diff_sets.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_XGB_grid.py

# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist_correct.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist_diff_sets.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist_test.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist_grid.py
conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist_diff_sets_evaluation.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/GBM_Hist_UQ.py


# conda run -n tensor python /home/botingl/machine\ learning\ copy/FCNN.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/CNN.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/CNN_paper.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/Linear.py
# conda run -n tensor python /home/botingl/machine\ learning\ copy/CNN_paper_modify.py
