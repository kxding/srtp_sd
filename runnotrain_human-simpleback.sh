cuda_device="2"
image_dir1="12-21-human-simpleback-withnorefcfg"
image_dir2="12-21-human-simpleback-withrefcfg"
image_dir3="12-21-human-simpleback-withrefcfg-directinversion"
image_dir4="12-21-human-simpleback-withnorefcfg-directinversion"
model_path="/data1/chenxuan/model"
# without cfg
emdedding_path1="./output/human-simpleback_12_20_2023_0023/"
# with cfg
emdedding_path2="./output/human-simpleback_12_20_2023_0036/"
initializer_token_my="human"

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/human-simpleback" \
                    --content_image_dir="./samples/base-human"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir1}" \
                    --cross_attention_injection_ratio=$ratio \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings
done

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/human-simpleback" \
                    --content_image_dir="./samples/base-human"  \
                    --src_embedding_path="${emdedding_path1}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir4}" \
                    --cross_attention_injection_ratio=$ratio \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion
done

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/human-simpleback" \
                    --content_image_dir="./samples/base-human"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=$ratio \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings
done

                    
for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./samples/human-simpleback" \
                    --content_image_dir="./samples/base-human"  \
                    --src_embedding_path="${emdedding_path2}" \
                    --initializer_token="${initializer_token_my}" \
                    --pretrained_model_name_or_path="${model_path}" \
                    --output_image_path="./output_images/${image_dir3}" \
                    --cross_attention_injection_ratio=$ratio \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_prev_embeddings --use_direct_inversion
done

