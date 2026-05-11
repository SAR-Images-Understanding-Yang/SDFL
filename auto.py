import os
import time


# SAMPLE
# for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#     for status in ["train", "analysis"]:
#         os.system("python ./SDFL.py"
#             " /home/xyang/datasets/SampleDataset"
#             " -d SAMPLE"
#             " -s S"
#             " -t R"
#             " -a resnet18"
#             " -b 36"
#             " -j 0"
#             " --epochs 20"
#             " --seed {}"
#             " --log /home/xyang/code/SARCDG_results/SAMPLE/SDFL/{}" 
#             " --phase {}".format(seed, seed, status))


# for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#     for status in ["train", "analysis"]:
#         os.system("python ./SDFL.py"
#             " /home/xyang/datasets/S2M"
#             " -d S2M"
#             " -s S"
#             " -t R"
#             " -a resnet18"
#             " -b 36"
#             " -j 0"
#             " --epochs 20"
#             " --seed {}"
#             " --log /home/xyang/code/SARCDG_results/S2M/SDFL/{}" 
#             " --phase {}".format(seed, seed, status))


for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for status in ["train", "analysis"]:
        os.system("python ./SDFL.py"
            " /home/xyang/datasets/SimulatedSARShip"
            " -d SimulatedSARShip"
            " -s S"
            " -t R"
            " -a resnet18"
            " -b 36"
            " -j 0"
            " --epochs 20"
            " --seed {}"
            " --log /home/xyang/code/SARCDG_results/SimulatedSARShip/SDFL/{}" 
            " --phase {}".format(seed, seed, status))
