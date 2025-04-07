#!/bin/bash


montepy=/Users/zhiyulu/Documents/Science/Cosmology_packages/montepython_public/montepython/Montepython.py

#source /exports/home/luzhiyu/data/Basic_download/anaconda3/etc/profile.d/conda.sh
#conda activate rsmarg


root_path="/Users/zhiyulu/Documents/Github/montepython_likelihood/Planck_PR4_lensing/"
cd $root_path

fold_name="compare_cobaya"
file_name="set"

echo "Current working directory: $(pwd)"

# Run the MPI program

rm -rf $fold_name/$file_name

python $montepy run -p $fold_name/${file_name}.param --conf ../class.conf -o $fold_name/$file_name -f 0

#mpirun -np 10 python $montepy run -p $fold_name/${file_name}.param --conf class.conf -N 5000000 -o $fold_name/chain -c $fold_name/xchain/chain.covmat


