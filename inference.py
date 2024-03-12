import torch
import os
import argparse

from run import parse_arguments, build_settings_str_from_args
from exp.exp_inference import Exp_Inference

# MODEL_NAME = "Exchange_96_96_Autoformer_custom_ftMS_sl60_ll30_pl30_dm32_nh8_el2_dl2_df32_fc3_ebtimeF_dtTrue_Exp_0"
# ckpt_file = os.path.join("checkpoints", MODEL_NAME, "checkpoint.pth")

args = parse_arguments()
# Add Sampling arguments
args_dict = vars(args)
args_dict["mc_dropout"] = True
args_dict["mc_samples"] = 50

settings = build_settings_str_from_args(args, 0)

exp = Exp_Inference(args)  # set experiments
exp.test(settings, test=1)
torch.cuda.empty_cache()
