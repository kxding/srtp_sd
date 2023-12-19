cuda_device="0"
image_dir1="12-11-tun-no-train"
image_dir2="12-11-tun-cfg-no-train"
image_dir3="12-11-tun-cfg+D-no-train"
image_dir4="12-11-tun-D-no-train"
model_path="/data1/chenxuan/model"
# with cfg
emdedding_path1="./output/tun_12_11_2023_0031/"
# without cfg
emdedding_path2="./output/tun_12_10_2023_2350/"
initializer_token_my="lion"
cross_attn=0.2
self_attn=0.9

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.6  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.6  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.4  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.4  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.8  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.6  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.6  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.4  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.4  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion