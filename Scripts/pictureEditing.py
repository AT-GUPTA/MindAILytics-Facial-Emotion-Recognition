
from PIL import Image
import os
from pathlib import Path
import time

paths = []

is_Testing: bool = True 
showAfter:bool = True
preferredSize = 640

if is_Testing:
    paths = ['./Image Samples/RD_1.jpg','./Image Samples/OSD_1.jpg', './Image Samples/SF-1001.jpg', './Image Samples/Surprised1.jpg']    
else:
    paths = ['./Datasets']

#Cropps the pictures that are larger than the preferredSize size to make them into square pictures.
#Resizes the pictures to the preferred size and saves them in the same folder as the original

def imgManipulation(path, is_Testing=True):
    testingName = ''
    img = Image.open(path)
    if img.size == (preferredSize,preferredSize):
        return
    if showAfter:
        img.show()
    widthToHeight = 0
    heightToWidth = 0
    left = 0
    top =  0
    right = 0
    bottom = 0
    resizedImg = cropped = img

    if img.width > img.height and (img.width > preferredSize or img.height > preferredSize):
        widthToHeight = img.width - img.height
        left = (widthToHeight / 2)
        top =  0
        right = (img.width - widthToHeight / 2)
        bottom = img.height
    if img.width < img.height and (img.width > preferredSize or img.height > preferredSize):
        heightToWidth = img.height - img.width
        left = 0
        top = (heightToWidth / 2)
        right = img.width
        bottom = (img.height) - (heightToWidth / 2)

    p = Path(path)

    if heightToWidth != 0 or widthToHeight != 0:
        cropped =  img.crop((left, top, right, bottom))
        if is_Testing:
            if showAfter:
                cropped.show()
            testingName = '_Cropped'
            cropped.save(str(p.parent.parent) + '\\' + str(p.parent) + '\\' + p.stem + testingName + p.suffix)
        else:
            cropped.save(path)
    resizedImg = cropped.resize((preferredSize,preferredSize))
    if is_Testing:
        testingName = '_Resized'
        if showAfter:
            resizedImg.show()
        resizedImg.save(str(p.parent.parent) + '\\' + str(p.parent) + '\\' + p.stem + testingName + p.suffix)
    else:
        resizedImg.save(path)




if is_Testing:
    for path in paths:
        imgManipulation(path, True)
else:
    for dataset in os.listdir(paths[0]):
        print(f'{dataset=}')
        dataset_path = os.path.join(paths[0], dataset)

        for _class in os.listdir(dataset_path):
            _class_path = os.path.join(dataset_path, _class)

            if os.path.isdir(_class_path):
                print(f'{_class_path=}')
                counter = 1
                for img_name in os.listdir(_class_path):
                    img_path = os.path.join(_class_path, img_name)
                    imgCount = len([name for name in os.listdir(_class_path) if os.path.isfile(img_path)])
                    print(f'\r' + "{:.0%}".format(counter/imgCount),end="\r")
                    if os.path.isfile(img_path):
                        time.sleep(0.01)
                        imgManipulation(img_path, False)
                        counter += 1
                print()



