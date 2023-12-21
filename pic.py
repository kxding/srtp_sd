
from PIL import Image  
import os  
  
# # 设置文件夹路径  
# folder_path1 = './output_images/12-21-human-complexback-withnorefcfg'
# folder_path2 = './output_images/12-21-human-complexback-withrefcfg'
# folder_path3 = './output_images/12-21-human-complexback-withrefcfg-directinversion'
# folder_path4 = './output_images/12-21-human-complexback-withnorefcfg-directinversion'

# folder_path5 = './output_images/12-20-human-simpleback-withnorefcfg-directinversion'
# folder_path6 = './output_images/12-20-human-simpleback-withrefcfg-directinversion'
# folder_path7 = './output_images/12-20-human-simpleback-withnorefcfg'
# folder_path8 = './output_images/12-20-human-simpleback-withrefcfg'

# folder_path9 = './output_images/12-21-bird-simpleback-withrefcfg'
# folder_path10 = './output_images/12-21-bird-simpleback-withnorefcfg'
# folder_path11 = './output_images/12-21-bird-simpleback-withrefcfg-directinversion'
# folder_path12 = './output_images/12-21-bird-simpleback-withnorefcfg-directinversion'

# folder_path13 = './output_images/12-21-bird-multiback-withrefcfg'
# folder_path14 = './output_images/12-21-bird-multiback-withnorefcfg'
# folder_path15 = './output_images/12-21-bird-multiback-withrefcfg-directinversion'
# folder_path16 = './output_images/12-21-bird-multiback-withnorefcfg-directinversion'

# folder_path17 = './output_images/12-21-lion-ce-withrefcfg'
# folder_path18 = './output_images/12-21-lion-ce-withnorefcfg'
# folder_path19 = './output_images/12-21-lion-ce-withrefcfg-directinversion'
# folder_path20 = './output_images/12-21-lion-ce-withnorefcfg-directinversion'

# folder_path21 = './output_images/12-21-lion-dark-withrefcfg'
# folder_path22 = './output_images/12-21-lion-dark-withnorefcfg'
# folder_path23 = './output_images/12-21-lion-dark-withrefcfg-directinversion'
# folder_path24 = './output_images/12-21-lion-dark-withnorefcfg-directinversion'





# folder_path_list = [folder_path6, folder_path7, folder_path8]
# for folder_path in folder_path_list:
#     # 获取文件夹中的所有图片文件名  
#     images = [file for file in os.listdir(folder_path) if file.endswith('.png')]  
#     # images.extend(images[:8])
#     # 自定义的排序函数，提取文件名中的数字作为排序关键字  
#     def extract_number(filename):  
#         return int(''.join(filter(str.isdigit, filename)))  
    
#     # 根据文件名中的数字排序图片文件  
#     sorted_images = sorted(images, key=extract_number)  

#     img_path = os.path.join(folder_path, sorted_images[0])
    
#     # 设置照片尺寸  
#     width, height = Image.open(img_path).size
    
#     # 创建一个新的空白图片  
#     new_image = Image.new('RGB', (width, height * len(sorted_images)))  
    
#     # 将每张图片粘贴到新的空白图片上  

#     for i, img in enumerate(sorted_images):  
#         print("img is: ", img)
#         img_path = os.path.join(folder_path, img)  
#         current_img = Image.open(img_path)  
#         new_image.paste(current_img, (0, i * height)) 


#     filename = os.path.basename(os.path.normpath(folder_path)) + ".png"  
#     output_path = os.path.join("paper_pic", filename)
#     new_image.save(output_path)

#将 "paper_pic"中的图片拼接成一张图
images = [file for file in os.listdir("paper_pic") if file.endswith('.png')]
def extract_number(filename):  
    return int(''.join(filter(str.isdigit, filename)))
sorted_images = sorted(images, key=extract_number)
img_path = os.path.join("paper_pic", sorted_images[0])
width, height = Image.open(img_path).size
new_image = Image.new('RGB', (width * len(sorted_images), height))
for i, img in enumerate(sorted_images):
    img_path = os.path.join("paper_pic", img)
    current_img = Image.open(img_path)
    new_image.paste(current_img, (i * width, 0))
new_image.save("paper_pic/all.png")
