# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:38:44 2017

@author: HSIN
"""

from PIL import Image
import numpy as np
import sys

im = Image.open(sys.argv[1])
im_modified = Image.open(sys.argv[2])

im_np_data = np.array(im.getdata())
im_modified_np_data = np.array(im_modified.getdata())
im_diff = im_np_data - im_modified_np_data

ans_two = Image.new('RGBA', im.size, color = None)

for i in range(len(im_np_data)):
    if not np.array_equal(im_diff[i], [0, 0, 0, 0]):
        ans_two.putpixel((i%im.size[0], int(i/im.size[0])), tuple(im_modified_np_data[i]))

ans_two.save('ans_two.png')
        
        