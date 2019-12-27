python svm_multiclass.py --classes=3 --C=0.01 --kernel=poly --kernel_degree=3 --test_size=339 --num_passes=5 --kernel_gamma=1 --seed=42 --tolerance=0.0001
echo new
python svm_multiclass.py --classes=3 --C=3 --kernel=poly --kernel_degree=3 --test_size=339 --num_passes=5 --kernel_gamma=1 --seed=42 --tolerance=0.0001
echo new
python svm_multiclass.py --classes=3 --C=2 --kernel=poly --kernel_degree=3 --test_size=339 --num_passes=5 --kernel_gamma=1 --seed=42 --tolerance=0.0001
echo new
python svm_multiclass.py --classes=3 --C=1 --kernel=poly --kernel_degree=3 --test_size=339 --num_passes=5 --kernel_gamma=1 --seed=42 --tolerance=0.0001
echo new
python svm_multiclass.py --classes=5 --C=1 --kernel=poly --kernel_degree=3 --kernel_gamma=1 --test_size=701 --num_passes=5 --seed=42 --tolerance=0.0001
echo new
python svm_multiclass.py --classes=4 --C=2 --kernel=poly --kernel_degree=4 --kernel_gamma=1 --test_size=640 --num_passes=5 --seed=42 --tolerance=0.0001
echo new
