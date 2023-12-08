# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100
                  

# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/che/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100
                  
# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/dra/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100
                  
# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/tun/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100

CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
                    --concept_image_dir="./examples/jieni/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/home/dingkaixin/s" \
                    --output_image_path="./output_images/12-6-6" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg
                  

CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
                    --concept_image_dir="./examples/che/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/12-6-6" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg
                  
CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
                    --concept_image_dir="./examples/dra/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/12-6-6" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg
                  
CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
                    --concept_image_dir="./examples/tun/" \
                    --content_image_dir="./examples/lion/"  \
                    --initializer_token="pokemon" \
                    --pretrained_model_name_or_path="/data1/chenxuan/model" \
                    --output_image_path="./output_images/12-6-6" \
                    --cross_attention_injection_ratio=0.05 \
                    --self_attention_injection_ratio=0.95  \
                    --guidance_scale_train_src=7.5 \
                    --guidance_scale_train_ref=7.5 \
                    --guidance_scale_gen=7.5 \
                    --only_save_embeds \
                    --use_l1 --max_train_steps=100 --use_ref_cfg

# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/jieni/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  

# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/che/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/dra/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# CUDA_VISIBLE_DEVICES="1" accelerate launch main.py \
#                     --concept_image_dir="./examples/tun/" \
#                     --content_image_dir="./examples/lion/"  \
#                     --initializer_token="pokemon" \
#                     --pretrained_model_name_or_path="/data1/chenxuan/model" \
#                     --output_image_path="./output_images/12-6-2" \
#                     --cross_attention_injection_ratio=0.05 \
#                     --self_attention_injection_ratio=0.95  \
#                     --guidance_scale_train_src=7.5 \
#                     --guidance_scale_train_ref=7.5 \
#                     --guidance_scale_gen=7.5 \
#                     --only_save_embeds \
#                     --use_l1 --max_train_steps=100 --use_ref_cfg --use_direct_inversion
                  
# accelerate launch main.py --concept_image_dir="./example_try/concept2/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg

# accelerate launch main.py --concept_image_dir="./example_try/concept3/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg

# accelerate launch main.py --concept_image_dir="./example_try/concept4/" --content_image_dir="./example_try/content/"  --initializer_token="pokemon" --pretrained_model_name_or_path="/root/autodl-tmp/model" --cross_attention_injection_ratio=0.2 --self_attention_injection_ratio=0.9 --guidance_scale_train_ref=7.5 --only_save_embeds --use_l1 --max_train_steps=100 --use_ref_cfg
