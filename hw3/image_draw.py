from PIL import Image
import csv
import numpy as np

dataset_label = []
dataset_feature = []

f = open('train.csv', 'r')

reader = csv.reader(f)
f.readline()
count = 0

for line in f:
    print(count)
    count = count + 1
    
    the_label = int(line.split(',')[0])
    the_feature = line[line.find(',') + 1:].split(' ')
    the_feature = list(map(int, the_feature))
    
    
    dataset_label.append(the_label)
    dataset_feature.append(the_feature)
    
    
    
    im_np_data = the_feature
    im_size = (48,48)
    out = Image.new('RGBA', im_size, color = None)
    for i in range(len(im_np_data)):
        intense = im_np_data[i]
        out.putpixel((i%im_size[0], int(i/im_size[0])), 
                    (intense,intense,intense,255))

    out.save('img/'+str(count)+'.jpg')
    if(count == 100):
        break
                    
    


f.close()

