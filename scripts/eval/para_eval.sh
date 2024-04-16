
# #model_path
# output/alpaca_clean_svd_lora_merged
# output/alpaca_clean_original_lora_merged_alpha64
# output/alpaca_clean_original_lora_merged
# output/batch_size1_result/alpaca_clean_svd_lora_merged


# #output_root_path
# alpaca_clean_svd_lora_merged
# alpaca_clean_original_lora_merged_alpha64
# alpaca_clean_original_lora_merged
# batch_size1_result-alpaca_clean_svd_lora_merged
bash scripts/eval/tulu_template_model_eval_table1.sh alpaca_clean_svd_lora_merged 0  2>&1 |tee logs/alpaca_clean_svd_lora_merged_alpacafarm.log& 
bash scripts/eval/tulu_template_model_eval_table1.sh alpaca_clean_original_lora_merged_alpha64 1 2>&1 |tee logs/alpaca_clean_original_lora_merged_alpha64_alpacafarm.log& 
bash scripts/eval/tulu_template_model_eval_table1.sh alpaca_clean_original_lora_merged 2 2>&1 |tee logs/alpaca_clean_original_lora_merged_alpacafarm.log& 
bash scripts/eval/tulu_template_model_eval_table1.sh batch_size1_result/alpaca_clean_svd_lora_merged 3 2>&1 |tee logs/eval_batch_size1_result_alpacafarm.log& 