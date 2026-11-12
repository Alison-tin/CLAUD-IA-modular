import cv2
import numpy as np

def put(img,txt,x,y,escala=0.55,color=(255,255,255),grosor=1):
    cv2.putText(img,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,escala,color,grosor,cv2.LINE_AA)


def panel(img,x,y,w,h,color=(30,30,30),alpha=0.55):
    sub=img[y:y+h,x:x+w]
    rect=np.full_like(sub,color)
    cv2.addWeighted(rect,alpha,sub,1-alpha,0,sub)
    img[y:y+h,x:x+w]=sub


def barra_conf(img,x,y,w,val,color,h=8):
    cv2.rectangle(img,(x,y),(x+w,y+h),(60,60,60),-1)
    cv2.rectangle(img,(x,y),(x+int(w*val),y+h),color,-1)

