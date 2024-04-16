import torch
import os
import argparse

from run import parse_arguments, build_settings_str_from_args
from exp.exp_inference import Exp_Inference


def args_inference():
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


class AutoformerModel:
    def __init__(self, model_name, model_ckpt):
        pass


    def get_settings_from_ckpt(self, model_name):
        # setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)
        # return setting

        model_id, model, data, features, seq_len, label_len, pred_len, d_model, n_heads, e_layers, d_layers, d_ff, factor, embed, distil, des, ii = model_name.split("_")
        settings = {
            "model_id": model_id,
            "model": model,
            "data": data,
            "features": features,
            "seq_len": seq_len,
            "label_len": label_len,
            "pred_len": pred_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "e_layers": e_layers,
            "d_layers": d_layers,
            "d_ff": d_ff,
            "factor": factor,
            "embed": embed,
            "distil": distil,
            "des": des,
            "ii": ii}
        return settings

    # def load_model(self):



if __name__ == "__main__":
    # args_inference()

    MODEL_CKPT = "/home/matthias/Projects/Autoformer/checkpoints/BTC_Informer_ohlcv_ftS_sl60_ll60_pl3_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_'SIGNAL'_0/checkpoint.pth"
    m = AutoformerModel("Name", MODEL_CKPT)
    settings = m.get_settings_from_ckpt("BTC_Informer_ohlcv_ftS_sl60_ll60_pl3_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_'SIGNAL'_0")
    p = 1