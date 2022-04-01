# bash scripts/test_bert.sh eider test hoi

seed=66

ablation=$1
name=$2
coref_method=$3

load_path=chkpt/EIDER_bert_${ablation}_${name}_best.pt

evi_pred_file=evi_result_${ablation}_bert-base-cased.pkl

eval_mode=dev_only

ensemble_mode=2
# ensemble_ablation=evi_pred
ensemble_ablation=evi_rule

evi_eval_mode=none

echo ablation ${ablation}
echo ensemble ${ensemble_mode}, ${ensemble_ablation}
echo evi_eval_mode ${evi_eval_mode}

python train.py --data_dir ./dataset/docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed ${seed} \
--num_class 97 \
--evaluation_steps -1 \
--save_path chkpt/EIDER_bert_${ablation}_${name}_test.pt \
--ablation ${ablation} \
--name ${name} \
--feature_path saved_features \
--coref_method ${coref_method} \
--eval_mode ${eval_mode} \
--evi_eval_mode ${evi_eval_mode} \
--ensemble_mode ${ensemble_mode} \
--ensemble_ablation ${ensemble_ablation} \
--evi_pred_file ${evi_pred_file} \
--load_path ${load_path}
