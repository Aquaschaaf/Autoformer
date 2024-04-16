import torch
import os
import argparse

from run import parse_arguments, build_settings_str_from_args
from exp.exp_inference import Exp_Inference

N_MODEL_ITERATIONS = 1
SAMPLE_LIMIT = 25000  # None

args = parse_arguments()
# Add Sampling arguments
args_dict = vars(args)
args_dict["mc_dropout"] = True
args_dict["mc_samples"] = 50

out_dir_suffix = "_mcsampling{}".format(args_dict["mc_samples"]) if args_dict["mc_dropout"] else None

for ii in range(N_MODEL_ITERATIONS):
    settings = build_settings_str_from_args(args, ii)

    exp = Exp_Inference(args)  # set experiments
    exp.test(settings, sample_limit=SAMPLE_LIMIT, test=1, out_dir_suffix=out_dir_suffix)
    torch.cuda.empty_cache()
