@REM -------------------FIRST BATCH-------------------
@REM python experiment.py --name het --save_losses --heterogeneous
@REM python experiment.py --name modular --save_losses --modular_peaks
@REM python experiment.py --name individual --save_losses --individual
@REM python experiment.py --name het_modular --save_losses --heterogeneous --modular_peaks
@REM python experiment.py --name het_individual --save_losses --heterogeneous --individual
@REM python experiment.py --name modular_individual --save_losses --modular_peaks --individual
@REM python experiment.py --name het_modular_individual --save_losses --heterogeneous --modular_peaks --individual

@REM python experiment.py --name bs32 --save_losses --bs 32 --batches 100000
@REM python experiment.py --name bs128 --save_losses --bs 128

@REM python experiment.py --name gc_min30 --save_losses --gc_scale_min 30 --gc_scale_max 300
@REM python experiment.py --name gc_min60 --save_losses --gc_scale_min 60 --gc_scale_max 300
@REM python experiment.py --name gc_min30max200 --save_losses --gc_scale_min 30 --gc_scale_max 200
@REM python experiment.py --name gc_min30max100 --save_losses --gc_scale_min 30 --gc_scale_max 100
@REM python experiment.py --name gc_max200 --save_losses --gc_scale_min 90 --gc_scale_max 200
@REM python experiment.py --name gc_max150 --save_losses --gc_scale_min 90 --gc_scale_max 150

@REM python experiment.py --name 2modules100cells --save_losses --n_modules 2 --n_per_module 100
@REM python experiment.py --name 3modules100cells --save_losses --n_modules 3 --n_per_module 100
@REM python experiment.py --name 5modules100cells --save_losses --n_modules 5 --n_per_module 100
@REM python experiment.py --name 2modules256cells --save_losses --n_modules 2 --n_per_module 256
@REM python experiment.py --name 2modules576cells --save_losses --n_modules 2 --n_per_module 576

@REM python experiment.py --name 2fits --save_losses  --train_env2
@REM python experiment.py --name 2fits10000batches --save_losses  --train_env2 --batches_env2 10000
@REM python experiment.py --name 2fits1000batches --save_losses  --train_env2 --batches_env2 1000
@REM python experiment.py --name 2fits200batches --save_losses  --train_env2 --batches_env2 200


@REM -------------------CORRECTED EXPERIMENTS-------------------
@REM python experiment.py --name baseline --save_losses --heterogeneous

@REM python experiment.py --name hidden20 --save_losses --heterogeneous --actor_hidden 20
@REM python experiment.py --name hidden50 --save_losses --heterogeneous --actor_hidden 50
@REM python experiment.py --name hidden200 --save_losses --heterogeneous --actor_hidden 200
@REM python experiment.py --name hidden500 --save_losses --heterogeneous --actor_hidden 500

@REM python experiment.py --name l1_0 --save_losses --heterogeneous --wd_l1 0
@REM python experiment.py --name l1_01 --save_losses --heterogeneous --wd_l1 0.1
@REM python experiment.py --name l1_001 --save_losses --heterogeneous --wd_l1 0.01
@REM python experiment.py --name l1_0001 --save_losses --heterogeneous --wd_l1 0.001
@REM python experiment.py --name l1_00001 --save_losses --heterogeneous --wd_l1 0.0001
@REM python experiment.py --name l1_000001 --save_losses --heterogeneous --wd_l1 0.00001

python experiment.py --name hp_0 --save_losses --heterogeneous --hidden_penalty 0
python experiment.py --name hp_05 --save_losses --heterogeneous --hidden_penalty 0.5
python experiment.py --name hp_01 --save_losses --heterogeneous --hidden_penalty 0.1
python experiment.py --name hp_005 --save_losses --heterogeneous --hidden_penalty 0.05
python experiment.py --name hp_0005 --save_losses --heterogeneous --hidden_penalty 0.005
python experiment.py --name hp_0001 --save_losses --heterogeneous --hidden_penalty 0.001

@REM python experiment.py --name homo --save_losses

@REM python experiment.py --name 2modules484cells --save_losses --heterogeneous --n_modules 2 --n_per_module 484
@REM python experiment.py --name 3modules324cells --save_losses --heterogeneous --n_modules 3 --n_per_module 324
@REM python experiment.py --name 5modules196cells --save_losses --heterogeneous --n_modules 5 --n_per_module 196
python experiment.py --name 25cells --save_losses --heterogeneous --n_per_module 25
python experiment.py --name 49cells --save_losses --heterogeneous --n_per_module 49
python experiment.py --name 169cells --save_losses --heterogeneous --n_per_module 169

@REM python experiment.py --name 2fits --save_losses --heterogeneous --train_env2

pause