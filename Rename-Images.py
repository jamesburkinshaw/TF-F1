import os

dir = 'C:\\Users\\jzburkinshaw\\Documents\\F1-TF Images\\Monaco\\'
renamedDir = 'C:\\Users\\jzburkinshaw\\Documents\\F1-TF Images\\Monaco\\'
location = 'Monaco'

count = 0 
for file in os.listdir(dir):
    filename = os.fsdecode(file)
    print(file)
    src = dir + filename 
    dest = renamedDir + location + str(count) + '.png'
    os.rename(src, dest)
    count +=1
