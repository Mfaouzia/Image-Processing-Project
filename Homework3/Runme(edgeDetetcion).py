#For n>3 personal images, compare the edge detection results using Prewitt, Sobel, Canny, and FDoG, 
# #and analyze/explain the results. 
#Analyze the results and compare the pros and cons of the evaluated methods
#Explain why FDoG may generate lines of different widths. Can FDoG produce edge of width 1? If possible, how?
import cv2
import matplotlib.pyplot as plt
import numpy
from numpy.lib import RankWarning
from numpy.lib.function_base import select 

imgName = 'img1'
imgPath = './' + imgName +'.jpg'
sobelThreshold = 80
prewittThreshold = 15
#sobel 算子, 分成两个部分可以输出中间图像
# sx = -1 0 1
#      -2 0 2
#      -1 0 1
def __sobelNoThreshold(img):
    X = img.shape[0]
    Y = img.shape[1]
    img = (cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)).astype(numpy.int16)
    newImg = numpy.zeros((X,Y),dtype=numpy.uint8)
    sobelopy = lambda img,x,y: (img[x+1][y-1] + img[x+1][y+1] - img[x-1][y-1] - img[x-1][y+1] + 2*(img[x+1][y] - img[x-1][y]))
    sobelopx = lambda img,x,y: (img[x-1][y+1] + img[x+1][y+1] - img[x-1][y-1] - img[x+1][y-1] + 2*(img[x][y+1] - img[x][y-1]))
    for x in range(1,X + 1):
        for y in range(1,Y + 1):
            Gx = sobelopx(img,x,y)
            Gy = sobelopy(img,x,y)
            G  = numpy.sqrt(pow(Gx,2) + pow(Gy,2))
            newImg[x - 1][y - 1] = G if G <= 255 else 255
    return newImg

def sobel(img):
    img = __sobelNoThreshold(img)
    X = img.shape[0]
    Y = img.shape[1]
    for x in range(0,X):
        for y in range(0,Y):
            if(img[x][y] <= sobelThreshold): img[x][y] = numpy.uint8(0)
    return img

#Prewitt 算子
def __prewittNoThreshold(img):
    X = img.shape[0]
    Y = img.shape[1]
    img = (cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)).astype(numpy.int16)
    newImg = numpy.zeros((X,Y),dtype=numpy.uint8)
    prewittopy = lambda img,x,y: (img[x+1][y-1] + img[x+1][y+1] - img[x-1][y-1] - img[x-1][y+1] + (img[x+1][y] - img[x-1][y]))
    prewittopx = lambda img,x,y: (img[x-1][y+1] + img[x+1][y+1] - img[x-1][y-1] - img[x+1][y-1] + (img[x][y+1] - img[x][y-1]))
    for x in range(1,X + 1):
        for y in range(1,Y + 1):
            Gx = prewittopx(img,x,y)
            Gy = prewittopy(img,x,y)
            G  = max(Gx, Gy)
            newImg[x - 1][y - 1] = G if G <= 255 else 255
    return newImg

def prewitt(img):
    img = __sobelNoThreshold(img)
    X = img.shape[0]
    Y = img.shape[1]
    for x in range(0,X):
        for y in range(0,Y):
            if(img[x][y] <= prewittThreshold): img[x][y] = numpy.uint8(0)
    return img

#Canny算子
class Canny():
    #step1：我们需要对图像做平滑处理，这里使用一个Gauss滤波器。这里直接使用一个3*3的模板
    gaussTemplate =         [ [1,2,1],
                              [2,4,2],     
                              [1,2,1]]
    gaussCoe    =   1/16
    gaussopLine =   lambda img,x,y,a : img[x][y-1] * Canny.gaussTemplate[a][0] + img[x][y] * Canny.gaussTemplate[a][1] + img[x][y+1] * Canny.gaussTemplate[a][2]
    gaussop     =   lambda img,x,y :   Canny.gaussCoe * (Canny.gaussopLine(img,x-1,y,0) + Canny.gaussopLine(img,x,y,1) + Canny.gaussopLine(img,x+1,y,2))

    def __init__(self, img) -> None:
        self.img = img
        self.X = img.shape[0]
        self.Y = img.shape[1]
        self.dx = numpy.zeros([self.X,self.Y])
        self.dy = numpy.zeros([self.X,self.Y])
        self.d  = numpy.zeros([self.X,self.Y])
        
    def smooth(self, img):
        img = (cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)).astype(numpy.int16)
        newImg = numpy.zeros((self.X,self.Y),dtype=numpy.uint8)
        for x in range(1, self.X + 1):
            for y in range(1, self.Y + 1):
                G = Canny.gaussop(img,x,y)
                newImg[x -1 ][y - 1] = G if G <= 255 else 255
        return newImg

    #step2 求梯度
    def diff(self, img):
        img = (cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)).astype(numpy.int16)
        for i in range(1, self.X + 1):
            for j in range(1, self.Y + 1):   
                self.dx[i - 1,j - 1] = img[i, j+1] - img[i, j]
                self.dy[i - 1,j - 1] = img[i+1, j] - img[i, j]        
                self.d[i - 1,j - 1] = numpy.sqrt(numpy.square(self.dx[i -1,j - 1]) + numpy.square(self.dy[i - 1,j - 1])) 

    #step3 非极大值抑制
    def NMS(self):
        newImg = numpy.zeros((self.X,self.Y),dtype=numpy.uint8)
        for x in range(1, self.X -1):
            for y in range(1, self.Y - 1):
                if(self.dx[x][y] == 0): continue
                #xy方向上的梯度大小
                gx = self.dx[x][y]
                gy = self.dy[x][y]
                dTemp = self.d[x][y]
                if(abs(gy) > abs(gx)):
                    weight = abs(gx) / abs(gy)
                    g2 = self.d[x-1][y]
                    g4 = self.d[x+1][y]
                    if(gx*gy):
                        #g1 g2
                        #   C
                        #   g4 g3
                        g1 = self.d[x-1][y-1]
                        g3 = self.d[x+1][y+1]
                    else:
                        #   g2 g1
                        #   C
                        #g3 g4 
                        g1 = self.d[x-1][y+1]
                        g3 = self.d[x+1][y-1]
                else:
                    weight = abs(gy) / abs(gx)
                    g2 = self.d[x][y + 1]
                    g4 = self.d[x][y - 1]
                    if(gx*gy):
                        #g3   
                        #g4 C g2
                        #     g1 
                        g3 = self.d[x-1][y-1]
                        g1 = self.d[x+1][y+1]
                    else:
                        #     g1   
                        #g4 C g2
                        #g3 
                        g1 = self.d[x-1][y+1]
                        g3 = self.d[x+1][y-1]
                #g1g2,g3g4插值
                dTmep1 = weight*g1 + (1 -weight)*g2
                dTemp2 = weight*g3 + (1 -weight)*g4
                if(dTemp >= dTmep1 and dTemp > dTemp2):
                    newImg[x][y] = self.d[x][y]
        return newImg
    #step4 双阈值
    def threshold(self):
        self.diff(self.smooth(self.img))
        self.NMS()
        img = self.d
        newImg = numpy.zeros((self.X,self.Y),dtype=numpy.uint8)
        TH     = 0.2 * numpy.max((img))
        TL     = 0.15  * numpy.max((img))
        for x in range(1, self.X - 1):
            for y in range(1, self.Y - 1):
                if(img[x][y] > TH):
                    newImg[x][y] = 255
                    continue
                if(img[x][y] < TL):
                    newImg[x][y] = 0
                    continue
                s8 = [img[x-1][y-1],img[x-1][y],img[x-1][y-1],
                      img[x][y-1],img[x][y+1],
                      img[x+1][y-1],img[x+1][y],img[x+1][y+1]]
                if(numpy.max(s8) > TH):
                    newImg[x][y] = 255
        return newImg
                
if __name__ == "__main__":
    img = cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE,)
    print("imgName = ",imgName,"size = ",img.shape[0],"x",img.shape[1])
    c1 = Canny(img)
    imgSobel = sobel(img)
    imgPrewitt = prewitt(img)
    #cv2.imwrite('./'+ imgName +'_sobel.jpg',imgSobel)
    #cv2.imwrite('./'+ imgName +'_prewitt.jpg',imgPrewitt)
    #cv2.imwrite('./'+ imgName +'_smooth.jpg',imgSmooth)
    #cv2.imwrite('./'+ imgName +'_diff.jpg',c1.d)
    #cv2.imwrite('./'+ imgName +'_NMS.jpg',imgNMS)
    #cv2.imwrite('./'+ imgName +'_canny.jpg',c1.threshold())
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.subplot(222)
    plt.imshow(imgSobel, cmap='gray')
    plt.subplot(223)
    plt.imshow(imgPrewitt, cmap='gray')
    plt.subplot(224)
    plt.imshow(c1.threshold(), cmap='gray')
    plt.show()