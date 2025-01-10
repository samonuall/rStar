CUDA_VISIBLE_DEVICES=0 python run_src/do_generate.py \
    --dataset_name GSM8K \
    --test_json_filename test_all \
    --model_ckpt Qwen/Qwen2-0.5B-Instruct \
    --note default \
    --num_rollouts 2