cuda_device="0"
image_dir1="12-7-2"
image_dir2="12-7-2-cfg"
image_dir3="12-7-2-cfg+D"
cross_attn=0.05
self_attn=0.95
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
                     --self_attention_injection_ratio=0.2 \
                     --cross_attention_injection_ratio=0.9

    
CUDA_VISIBLE_DEVICE=1 nohup accelerate launch main.py \
                     --concept_image_dir="../examples/tun/" \
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
                     --self_attention_injection_ratio=0.2 \
                     --cross_attention_injection_ratio=0.9

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
                     --self_attention_injection_ratio=0.2 \
                     --cross_attention_injection_ratio=0.9