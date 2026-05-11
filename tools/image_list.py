'''
Generate image list for a dataset
用于给数据生成txt文件，txt文件中每行包含图片路径和对应的标签，格式如下：
/path/to/image1.jpg 0
'''

import os

def write_image_list(root_dir, output_file):
    with open(output_file, 'w') as f:
        label_mapping = {}
        label_counter = 0

        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.jpg'):
                    image_path = os.path.join(dirpath, filename)
                    label = os.path.basename(dirpath)

                    if label not in label_mapping:
                        label_mapping[label] = label_counter
                        label_counter += 1

                    f.write(f"{image_path} {label_mapping[label]}\n")

    print("Image list generated successfully!")

# Usage example

root_directory = r'/home/xyang/datasets/SimulatedSARShip/MLDG3'
output_txt = r'/home/xyang/datasets/SimulatedSARShip/MLDG3.txt'

write_image_list(root_directory, output_txt)