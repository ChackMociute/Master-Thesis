python experiment.py --name 1modules1024cells --save_losses --heterogeneous --n_modules 1 --n_per_module 1024 --gc_scale_min 260
python experiment.py --name 2modules484cells --save_losses --heterogeneous --n_modules 2 --n_per_module 484
python experiment.py --name 3modules324cells --save_losses --heterogeneous --n_modules 3 --n_per_module 324
python experiment.py --name 5modules196cells --save_losses --heterogeneous --n_modules 5 --n_per_module 196
python experiment.py --name 25cells --save_losses --heterogeneous --n_per_module 25
python experiment.py --name 49cells --save_losses --heterogeneous --n_per_module 49
python experiment.py --name 169cells --save_losses --heterogeneous --n_per_module 169

python experiment.py --name 1modules1024cells_homo --save_losses --n_modules 1 --n_per_module 1024 --gc_scale_min 260