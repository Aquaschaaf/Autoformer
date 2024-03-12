export CUDA_VISIBLE_DEVICES=0

source ./venv/bin/activate

# ======================================
# Multivariate -> Multivariate
# ======================================
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/btc_usd/ \
  --data_path btc_usd.csv \
  --model_id Exchange_96_96 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 50 \
  --do_predict

# ======================================
# Univariate -> Univariate
# =====================================
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_96_96 \
#  --model Autoformer \
#  --data custom \
#  --features S \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 1 \
#  --enc_in 1 \
#  --dec_in 1 \
#  --c_out 1 \
#  --des 'Exp' \
#  --itr 1 \
#  --train_epochs 50
#
# ======================================
# Multivariate -> Univariate - ?
# =====================================
#python -u run.py \
#  --is_training 1 \
#  --root_path ./dataset/exchange_rate/ \
#  --data_path exchange_rate.csv \
#  --model_id Exchange_96_96 \
#  --model Autoformer \
#  --data custom \
#  --features MS \
#  --seq_len 96 \
#  --label_len 48 \
#  --pred_len 96 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --factor 3 \
#  --enc_in 8 \
#  --dec_in 8 \
#  --c_out 1 \
#  --des 'MSExp' \
#  --itr 1 \
#  --train_epochs 50
