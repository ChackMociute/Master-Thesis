@REM python experiment.py --name baseline --save_losses --heterogeneous

@REM python experiment.py --name l1_0 --save_losses --heterogeneous --wd_l1 0
@REM python experiment.py --name l1_01 --save_losses --heterogeneous --wd_l1 0.1
@REM python experiment.py --name l1_001 --save_losses --heterogeneous --wd_l1 0.01
@REM python experiment.py --name l1_0001 --save_losses --heterogeneous --wd_l1 0.001
@REM python experiment.py --name l1_00001 --save_losses --heterogeneous --wd_l1 0.0001
@REM python experiment.py --name l1_000001 --save_losses --heterogeneous --wd_l1 0.00001

@REM python experiment.py --name hp_0 --save_losses --heterogeneous --hidden_penalty 0
@REM python experiment.py --name hp_05 --save_losses --heterogeneous --hidden_penalty 0.5
@REM python experiment.py --name hp_01 --save_losses --heterogeneous --hidden_penalty 0.1
@REM python experiment.py --name hp_005 --save_losses --heterogeneous --hidden_penalty 0.05
@REM python experiment.py --name hp_0005 --save_losses --heterogeneous --hidden_penalty 0.005
@REM python experiment.py --name hp_0001 --save_losses --heterogeneous --hidden_penalty 0.001

@REM python experiment.py --name homo --save_losses

@REM python experiment.py --name 2modules484cells --save_losses --heterogeneous --n_modules 2 --n_per_module 484
@REM python experiment.py --name 3modules324cells --save_losses --heterogeneous --n_modules 3 --n_per_module 324
@REM python experiment.py --name 5modules196cells --save_losses --heterogeneous --n_modules 5 --n_per_module 196
@REM python experiment.py --name 25cells --save_losses --heterogeneous --n_per_module 25
@REM python experiment.py --name 49cells --save_losses --heterogeneous --n_per_module 49
@REM python experiment.py --name 169cells --save_losses --heterogeneous --n_per_module 169

@REM python experiment.py --name hidden20 --save_losses --heterogeneous --actor_hidden 20
@REM python experiment.py --name hidden50 --save_losses --heterogeneous --actor_hidden 50
@REM python experiment.py --name hidden200 --save_losses --heterogeneous --actor_hidden 200
@REM python experiment.py --name hidden500 --save_losses --heterogeneous --actor_hidden 500

@REM python exp_two_envs.py --name 2fits --save_losses --heterogeneous --train_env2

pause