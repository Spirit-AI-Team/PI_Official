export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

python scripts/serve_policy.py --env SPI0_ARX_FULL2 --default_prompt='Flatten the shirt'


