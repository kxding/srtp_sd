cuda_device="2"
image_dir1="12-7-4"
image_dir2="12-7-4-cfg"
image_dir3="12-7-4-cfg+D"
cross_attn=0.2
self_attn=0.9

 CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                     --concept_image_dir="./examples/jieni/" \
                     --content_image_dir="./examples/lion/"  \
                     --initializer_token="pokemon" \
                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
                     --output_image_path="./output_images/${image_dir1}" \
                     --cross_attention_injection_ratio=${cross_attn} \
                     --self_attention_injection_ratio=${self_attn}  \
                     --guidance_scale_train_src=7.5 \
                     --guidance_scale_train_ref=7.5 \
                     --guidance_scale_gen=7.5 \
                     --only_save_embeds \
                     --use_l1 --max_train_steps=100
                  

 CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                     --concept_image_dir="./examples/che/" \
                     --content_image_dir="./examples/lion/"  \
                     --initializer_token="pokemon" \
                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
                     --output_image_path="./output_images/${image_dir1}" \
                     --cross_attention_injection_ratio=${cross_attn} \
                     --self_attention_injection_ratio=${self_attn}  \
                     --guidance_scale_train_src=7.5 \
                     --guidance_scale_train_ref=7.5 \
                     --guidance_scale_gen=7.5 \
                     --only_save_embeds \
                     --use_l1 --max_train_steps=100
                  
 CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                     --concept_image_dir="./examples/dra/" \
                     --content_image_dir="./examples/lion/"  \
                     --initializer_token="pokemon" \
                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
                     --output_image_path="./output_images/${image_dir1}" \
                     --cross_attention_injection_ratio=${cross_attn} \
                     --self_attention_injection_ratio=${self_attn}  \
                     --guidance_scale_train_src=7.5 \
                     --guidance_scale_train_ref=7.5 \
                     --guidance_scale_gen=7.5 \
                     --only_save_embeds \
                     --use_l1 --max_train_steps=100
                  
 CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                     --concept_image_dir="./examples/tun/" \
                     --content_image_dir="./examples/lion/"  \
                     --initializer_token="pokemon" \
                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
                     --output_image_path="./output_images/${image_dir1}" \
                     --cross_attention_injection_ratio=${cross_attn} \
                     --self_attention_injection_ratio=${self_attn}  \
                     --guidance_scale_train_src=7.5 \
                     --guidance_scale_train_ref=7.5 \
                     --guidance_scale_gen=7.5 \
                     --only_save_embeds \
                     --use_l1 --max_train_steps=100

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/jieni/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=${cross_attn} \
                    --self_attention_injection_ratio=${self_attn}  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg
                  

CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/che/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=${cross_attn} \
                    --self_attention_injection_ratio=${self_attn}  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg
                  
CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/dra/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=${cross_attn} \
                    --self_attention_injection_ratio=${self_attn}  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg
                  
CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/${image_dir2}" \
                    --cross_attention_injection_ratio=${cross_attn} \
                    --self_attention_injection_ratio=${self_attn}  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg

# CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/${image_dir3}" \
#                     --cross_attention_injection_ratio=${cross_attn} \
#                     --self_attention_injection_ratio=${self_attn}  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
#
#
# CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
#                     --concept_image_dir="./examples/che/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/${image_dir3}" \
#                     --cross_attention_injection_ratio=${cross_attn} \
#                     --self_attention_injection_ratio=${self_attn}  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
#
# CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
#                     --concept_image_dir="./examples/dra/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/${image_dir3}" \
#                     --cross_attention_injection_ratio=${cross_attn} \
#                     --self_attention_injection_ratio=${self_attn}  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
#
# CUDA_VISIBLE_DEVICES=${cuda_device} accelerate launch main.py \
#                     --concept_image_dir="./examples/tun/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/${image_dir3}" \
#                     --cross_attention_injection_ratio=${cross_attn} \
#                     --self_attention_injection_ratio=${self_attn}  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# accelerate launch main.py --concept_image_dir="./example_try/concept2/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg

# accelerate launch main.py --concept_image_dir="./example_try/concept3/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg

# accelerate launch main.py --concept_image_dir="./example_try/concept4/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg
