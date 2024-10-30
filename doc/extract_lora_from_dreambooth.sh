python sd-scripts/networks/extract_lora_from_models.py \
  --sdxl \
 --model_org ./sd-models/sd_xl_base_1.0.safetensors \
 --model_tuned ./output/db/db_4496_0925_2.safetensors \
 --save_to lora_4496_0925_2_dim128.safetensors \
 --dim 128 \
 --conv_dim 128 \
 --save_precision fp16