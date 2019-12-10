python smo_algorithm.py --C 1 --kernel linear --tolerance 0.0001 --examples 160 --num_passes 5 --seed 007 --test_ratio 0.625
echo new
python smo_algorithm.py --C 1 --kernel rbf --kernel_gamma=1 --tolerance 0.0001 --examples 160 --num_passes 5 --seed 007 --test_ratio 0.625
echo new
python smo_algorithm.py --C 1 --kernel rbf --kernel_gamma=0.1 --tolerance 0.0001 --examples 160 --num_passes 5 --seed 007 --test_ratio 0.625
echo new
python smo_algorithm.py --C 1 --kernel poly --kernel_degree=3 --kernel_gamma=1 --tolerance 0.0001 --examples 160 --num_passes 5 --seed 007 --test_ratio 0.625
echo new
python smo_algorithm.py --C 5 --kernel poly --kernel_degree=3 --kernel_gamma=1 --tolerance 0.0001 --examples 160 --num_passes 5 --seed 007 --test_ratio 0.625
