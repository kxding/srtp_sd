cuda_device="0"
image_dir1="12-18-lion-zheng-withnorefcfg"
image_dir2="12-18-lion-zheng-withrefcfg"
image_dir3="12-18-lion-zheng-withrefcfg-directinversion"
model_path="/data1/chenxuan/model"
# without cfg
emdedding_path1="./output/lion-zheng_12_17_2023_2309/"
# with cfg
emdedding_path2="./output/lion-zheng_12_17_2023_2323/"
initializer_token_my="lion"

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
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
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
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
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.85 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.8 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.75 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.7 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.65 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.55 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.5 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.45 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.35 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.3 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.15 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 




CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
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
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
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
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.85 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.8 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.75 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.7 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.65 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.55 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.5 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.45 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.35 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.3 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.15 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings 




CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.9 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --no_controller --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
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
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
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
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.85 \
                    --self_attention_injection_ratio=0.2  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.8 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.75 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion
 
CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.7 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.65 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.6 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.55 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.5 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.45 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.4 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.35 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.3 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.25 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion
 
CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.15 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.1 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings  --use_direct_inversion

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/lion-zheng" \
                    --content_image_dir="./samples/base-lion"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.2 \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion

