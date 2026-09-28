"""
run_all.py
==========
Convenience wrapper: train the score model, then run the reconstruction

"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script):
    print("")
    print("########## " + script + " ##########")
    r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(script + " failed with code " + str(r.returncode))


if __name__ == "__main__":
    run("train_score.py")
    run("reconstruct_pet_dds.py")
    print("")
    print("DONE")
