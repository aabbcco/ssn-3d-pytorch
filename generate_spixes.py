import os
import argparse

argf = argparse.ArgumentParser()
argf.add_argument('--dd',type=int,help="you know how it uses")
argg = argf.parse_args()

commands = ["--weight log1/bset_model_sp_test.pth --nspix "," --weight log/bset_model_uc;_test.pth --nspix ","--nspix "]
commands2 = ["spix","orig","single"]

for i in range(1, 11):
    command = "conda activate aabbcco-torch ; python eval_on_test_set.py "+commands[argg.dd]+"{:d} -d results_".format(i*100)+commands2[argg.dd]+"_{:d}".format(i*100)
    os.system(command)