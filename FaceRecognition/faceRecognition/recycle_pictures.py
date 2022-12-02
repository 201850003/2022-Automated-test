import os
import string
import numpy
import cv2

#对一张照片进行处理
def processPictures(pictureName,picturePath,type):
    img = cv2.imread(picturePath)
    #img_flip = cv2.flip(img, 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


#识别一张照片并进行处理
def readFace(pictureName,picturePath,objectPath,type):
        if picturePath[-4:] == type:
            print('读入照片'+pictureName)
            img = processPictures(pictureName,picturePath,type)
            face_cascade = cv2.CascadeClassifier(
                'D://work/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                f = cv2.resize(img[y:(y + h), x:(x + w)], (200, 200))
                cv2.imwrite(objectPath + os.sep + '%s' % pictureName, f)




def readAllFaces(sourcePath,objectPath,type):
    dirs = os.listdir(sourcePath)
    for d in dirs:
        filePath = sourcePath + '/' + d
        if os.path.isdir(filePath):
            readAllFaces(filePath, objectPath,type)
            #是文件
        else:
            readFace(d,filePath,objectPath,type)

def readFiftyImgs(sourcePath,objectPath,type):
    dirs = os.listdir(sourcePath)
    for d in dirs:
        filePath = sourcePath + '/' + d
        imgs = os.listdir(filePath)
        count = 0
        for i in imgs:
            if count<50:
                imgPath = filePath + '/' +i
                img = cv2.imread(imgPath)
                cv2.imwrite(objectPath + os.sep + '%s' % i,img)
                count+=1


if __name__ == '__main__':
    sourcePath='C://Users/BYQ/Desktop/pre-tool/pictures/105_classes_pins_dataset/'
    objectPath='C://Users/BYQ/Desktop/test/t'
    readAllFaces(sourcePath,objectPath,'.jpg')
    readFiftyImgs(sourcePath,objectPath,'.jpg')
