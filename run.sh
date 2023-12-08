# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.9  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --no_controller
                  

# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/che/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.9  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --no_controller
                  
# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/dra/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.9  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --no_controller
                  
# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/tun/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.9  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --no_controller

# nohup \
accelerate launch main.py \
                    --concept_image_dir="./examples/jieni/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/kaixin/s" \
                    --output_image_path="./output_images/12-6-5" \
                    --cross_attention_injection_ratio=0.2 \
                    --self_attention_injection_ratio=0.9  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=10 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion

# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-5" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.9  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=12 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  

# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-5" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.1  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=10 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg 
                  
# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-5" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.2  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=10 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-5" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.3  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=10 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion

# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.9  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  

# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.7  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.8  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# CUDA_VISIBLE_DEVICES="0" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/home/dingkaixin/s" \
#                     --output_image_path="./output_images/12-6-1" \
#                     --cross_attention_injection_ratio=0.2 \
#                     --self_attention_injection_ratio=0.6  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# # accelerate launch main.py --concept_image_dir="./example_try/concept2/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg

# # accelerate launch main.py --concept_image_dir="./example_try/concept3/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg

# # accelerate launch main.py --concept_image_dir="./example_try/concept4/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg
