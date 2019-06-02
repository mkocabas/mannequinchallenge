import os
import sys

folder = sys.argv[1]
output_f = sys.argv[2]

image_list = [os.path.join(folder, x)
              for x in os.listdir(folder)
              if x.endswith('.jpg') or x.endswith('.png')]

with open(output_f, 'w') as f:
    for item in image_list:
        f.write(f'{item}\n')