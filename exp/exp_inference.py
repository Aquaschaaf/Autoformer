import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from exp.exp_main import Exp_Main
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Inference(Exp_Main):
    def __init__(self, args):
        super(Exp_Inference, self).__init__(args)

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y


    def test(self, setting, sample_limit=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        stds = []
        trues = []
        final_inputs = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.args.mc_dropout:
            def get_layers(model: torch.nn.Module):
                children = list(model.children())
                return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]

            layers = get_layers(self.model)
            for layer in layers:
                if isinstance(layer, nn.Dropout):
                    layer.train()

        num_samples = self.args.mc_samples if self.args.mc_dropout else 1

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if sample_limit is not None:
                    if i >= sample_limit/batch_x.shape[0]:
                        break
                if i % 1000 == 0:
                    s_num = 1 if sample_limit is None else sample_limit/batch_x.shape[0]
                    max_samples = np.max([len(test_loader), s_num])
                    print("Processing test sample {}/{}".format(i, max_samples))

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
Matthi
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                all_outputs = []
                for n_sample in range(num_samples):
                    outputs, batch_y_out = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    all_outputs.append(outputs)

                all_outputs = torch.stack(all_outputs)
                mean = torch.mean(all_outputs, dim=0)
                std = torch.std(all_outputs, dim=0)

                input = batch_x.detach().cpu().numpy()
                mean = mean.detach().cpu().numpy()
                std = std.detach().cpu().numpy()
                # outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y_out.detach().cpu().numpy()

                pred = mean  # outputs.detach().cpu().numpy()  # .squeeze()
                std = std  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                final_input = input[:, -1, -1]

                preds.append(pred)
                stds.append(std)
                trues.append(true)
                final_inputs.append(final_input)
                if i % 20 == 0:
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    _std = np.concatenate((np.zeros(len(input[0, :, -1])), std[0, :, -1]), axis=0)
                    visual(gt, pd, std=_std, name=os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        stds = np.concatenate(stds, axis=0)
        trues = np.concatenate(trues, axis=0)
        final_inputs = np.concatenate(final_inputs, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        stds = stds.reshape(-1, stds.shape[-2], stds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        final_inputs = final_inputs.reshape(-1)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'stds.npy', stds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'final_inputs.npy', final_inputs)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
