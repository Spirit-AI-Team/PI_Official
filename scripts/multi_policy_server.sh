
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=5
POLICY1_CONFIG=spi0_aloha_finetune_full_15steps
POLICY1_DIR=/pfstem/wenxuan/resources/checkpoints/spi0_aloha_finetune_full_15steps/spi0_aloha_15steps_ftfull_0304/10000
POLICY1_PROMPT="Flatten the shirt"

POLICY2_CONFIG=spi0_aloha_finetune_lora_15steps
POLICY2_DIR=/pfstem/wenxuan/resources/checkpoints/spi0_aloha_finetune_lora_15steps/spi0_aloha_15steps_ftlora_0304/10000
POLICY2_PROMPT="Fold the shirt"

python scripts/serve_multipolicy.py --default_prompt1="${POLICY1_PROMPT}" --default_prompt2="${POLICY2_PROMPT}" policy1:checkpoint --policy1.config=$POLICY1_CONFIG --policy1.dir=$POLICY1_DIR  policy2:checkpoint --policy2.config=$POLICY2_CONFIG --policy2.dir=$POLICY2_DIR 




