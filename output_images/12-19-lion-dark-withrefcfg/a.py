# from PIL import Image
# import os
# folder_path = 'VCT_v3/output_images/12-19-lion-dark-withnorefcfg'
# images = [file in file in os.listdir(folder_path) if file.endswith('.png')]
# def extract_number(f):
#     s = f.split('_')
#     return int(s[10])
# images = sorted(images, key=extract_number)
# width, height = Image.open(images[0]).size
# nwe_image = Image.new('RGB', (width, height * len(images)))

# for i, img in enumerate(images):
#     im = Image.open(img)
#     nwe_image.paste(im, (0, i * height))

# nwe_image.save('VCT_v3/output_images/12-19-lion-dark-withnorefcfg/a.png')

from PIL import Image  
import os  
  
# 设置文件夹路径  
folder_path = './'
  
# 获取文件夹中的所有图片文件名  
images = [file for file in os.listdir(folder_path) if file.endswith('.png')]  
  
# 自定义的排序函数，提取文件名中的数字作为排序关键字  
def extract_number(filename):  
    return int(''.join(filter(str.isdigit, filename)))  
  
# 根据文件名中的数字排序图片文件  
sorted_images = sorted(images, key=extract_number)  

  
# 设置照片尺寸  
width, height = Image.open(sorted_images[0]).size
  
# 创建一个新的空白图片  
new_image = Image.new('RGB', (width, height * len(sorted_images)))  
  
# 将每张图片粘贴到新的空白图片上  
for i, img in enumerate(sorted_images):  
    print("img is: ", img)
    img_path = os.path.join(folder_path, img)  
    current_img = Image.open(img_path)  
    new_image.paste(current_img, (0, i * height))  
  
# 保存新的图片  
new_image.save("sorted_combined_image.jpg")  