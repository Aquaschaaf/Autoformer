import torch
import os
import argparse

from run import parse_arguments, build_settings_str_from_args
from exp.exp_inference import Exp_Inference

N_MODEL_ITERATIONS = 10
SAMPLE_LIMIT = 25000  # None

args = parse_arguments()
# Add Sampling arguments
args_dict = vars(args)
args_dict["mc_dropout"] = True
args_dict["mc_samples"] = 50

for ii in range(N_MODEL_ITERATIONS):
    settings = build_settings_str_from_args(args, ii)

    exp = Exp_Inference(args)  # set experiments
    exp.test(settings, sample_limit=SAMPLE_LIMIT, test=1)
    torch.cuda.empty_cache()
