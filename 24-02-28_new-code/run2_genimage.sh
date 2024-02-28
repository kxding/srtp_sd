cuda_device="0"
image_dir1="12-7-2"
image_dir2="12-7-2-cfg"
image_dir3="12-7-2-cfg+D"
self_attn=0.9

# cross_attn=0.2
# CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
#                         --concept_image_dir="../examples/che/" \
#                         --content_image_dir="../examples/lion/"  \
#                         --initializer_token="pokemon" \
#                         --pretrained_model_name_or_path="/home/asc2/dkx/s" \
#                         --output_image_path="./output_images/${image_dir1}" \
#                         --cross_attention_injection_ratio=${cross_attn} \
#                         --self_attention_injection_ratio=${self_attn}  \
#                         --guidance_scale_train_src=7.5 \
#                         --guidance_scale_train_ref=7.5 \
#                         --guidance_scale_gen=7.5 \
#                         --only_save_embeds \
#                         --use_l1 \
#                         --max_train_step=100 \
#                         --no_controller \
#                         --src_embedding_path="../model/tun_02_26_2024_1753" \
#                         --use_prev_embeddings 
for cross_attn in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8  
do
    CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
                        --concept_image_dir="../examples/lion/" \
                        --content_image_dir="../examples/tun/"  \
                        --initializer_token="pokemon" \
                        --pretrained_model_name_or_path="/home/asc2/dkx/s" \
                        --output_image_path="./output_images/${image_dir1}" \
                        --cross_attention_injection_ratio=${cross_attn} \
                        --self_attention_injection_ratio=${self_attn}  \
                        --guidance_scale_train_src=7.5 \
                        --guidance_scale_train_ref=7.5 \
                        --guidance_scale_gen=7.5 \
                        --only_save_embeds \
                        --use_l1 \
                        --max_train_step=100 \
                        --no_controller \
                        --src_embedding_path="./output/lion_02_28_2024_1918" \
                        --use_prev_embeddings 

    CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
                        --concept_image_dir="../examples/che/" \
                        --content_image_dir="../examples/tun/"  \
                        --initializer_token="pokemon" \
                        --pretrained_model_name_or_path="/home/asc2/dkx/s" \
                        --output_image_path="./output_images/${image_dir1}" \
                        --cross_attention_injection_ratio=${cross_attn} \
                        --self_attention_injection_ratio=${self_attn}  \
                        --guidance_scale_train_src=7.5 \
                        --guidance_scale_train_ref=7.5 \
                        --guidance_scale_gen=7.5 \
                        --only_save_embeds \
                        --use_l1 \
                        --max_train_step=100 \
                        --no_controller \
                        --src_embedding_path="./output/che_02_28_2024_1931" \
                        --use_prev_embeddings 

     CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
                        --concept_image_dir="../examples/che/" \
                        --content_image_dir="../examples/lion/"  \
                        --initializer_token="pokemon" \
                        --pretrained_model_name_or_path="/home/asc2/dkx/s" \
                        --output_image_path="./output_images/${image_dir1}" \
                        --cross_attention_injection_ratio=${cross_attn} \
                        --self_attention_injection_ratio=${self_attn}  \
                        --guidance_scale_train_src=7.5 \
                        --guidance_scale_train_ref=7.5 \
                        --guidance_scale_gen=7.5 \
                        --only_save_embeds \
                        --use_l1 \
                        --max_train_step=100 \
                        --no_controller \
                        --src_embedding_path="./output/che_02_28_2024_1944" \
                        --use_prev_embeddings                        

    CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
                        --concept_image_dir="../examples/che/" \
                        --content_image_dir="../examples/tun/"  \
                        --initializer_token="pokemon" \
                        --pretrained_model_name_or_path="/home/asc2/dkx/s" \
                        --output_image_path="./output_images/${image_dir1}" \
                        --cross_attention_injection_ratio=${cross_attn} \
                        --self_attention_injection_ratio=${self_attn}  \
                        --guidance_scale_train_src=7.5 \
                        --guidance_scale_train_ref=7.5 \
                        --guidance_scale_gen=7.5 \
                        --only_save_embeds \
                        --use_l1 \
                        --max_train_step=100 \
                        --no_controller \
                        --src_embedding_path="./output/che_02_28_2024_1957" \
                        --use_prev_embeddings 

done

# for cross_attn in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8  
# do
#     CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
#                         --concept_image_dir="../examples/che/" \
#                         --content_image_dir="../examples/tun/"  \
#                         --initializer_token="pokemon" \
#                         --pretrained_model_name_or_path="/home/asc2/dkx/s" \
#                         --output_image_path="./output_images/${image_dir1}" \
#                         --cross_attention_injection_ratio=${cross_attn} \
#                         --self_attention_injection_ratio=${self_attn}  \
#                         --guidance_scale_train_src=7.5 \
#                         --guidance_scale_train_ref=7.5 \
#                         --guidance_scale_gen=7.5 \
#                         --only_save_embeds \
#                         --use_l1 \
#                         --max_train_step=100 \
#                         --no_controller \
#                         --src_embedding_path="../model/tun_02_26_2024_1753" \
#                         --use_prev_embeddings 
# done                

# for cross_attn in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8  
# do
#     CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
#                         --concept_image_dir="../examples/che/" \
#                         --content_image_dir="../examples/lion/"  \
#                         --initializer_token="pokemon" \
#                         --pretrained_model_name_or_path="/home/asc2/dkx/s" \
#                         --output_image_path="./output_images/${image_dir1}" \
#                         --cross_attention_injection_ratio=${cross_attn} \
#                         --self_attention_injection_ratio=${self_attn}  \
#                         --guidance_scale_train_src=7.5 \
#                         --guidance_scale_train_ref=7.5 \
#                         --guidance_scale_gen=7.5 \
#                         --only_save_embeds \
#                         --use_l1 \
#                         --max_train_step=100 \
#                         --no_controller \
#                         --src_embedding_path="../model/lion_02_26_2024_1806" \
#                         --use_prev_embeddings 
# done