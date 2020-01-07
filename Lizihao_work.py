#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFile


# In[10]:


def image_crop(img_path,start_x,stop_x,start_y,stop_y):#图片裁剪
    img=cv2.imread(img_path)
    print("裁剪前的尺寸大小")
    print(img.shape)
    cv2.imshow('before',img)
    cropped=img[start_x:stop_x,start_y:stop_y]
    print("裁剪后的尺寸大小")
    print(cropped.shape)
    cv2.imshow('after',cropped)
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()


# In[3]:


image_crop('lenna.jpg',0,250,100,300)


# In[31]:


def color_shift(img_path):#图片色彩变化，分别是亮度、对比度、锐度
    print("图片色彩变化")
    image=Image.open(img_path)
    random_factor=np.random.randint(0,31)/10.
    color_image=ImageEnhance.Color(image).enhance(random_factor)
    color_image.show()
    random_factor=np.random.randint(10,21)/10.
    brightness_image=ImageEnhance.Brightness(color_image).enhance(random_factor)
    brightness_image.show()
    random_factor=np.random.randint(10,21)/10.
    contrast_image=ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    contrast_image.show()
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    sharp_image=ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 
    sharp_image.show()              
                
    


# In[32]:


color_shift('lenna.jpg')


# In[43]:


def image_rotation(img_path,angle):#图像旋转角度变化
    print("图像旋转角度变化")
    image=Image.open(img_path)
    img_rotation=image.rotate(angle,Image.BICUBIC)
    img_rotation.show()


# In[44]:


image_rotation('lenna.jpg',50)


# In[2]:


def image_perspective(img_path,nw_x,nw_y,ne_x,ne_y,sw_x,sw_y,se_x,se_y):#图像透视变化
    print("图像透视变化")
    image=cv2.imread(img_path)
    wigth,height=image.shape[:2]
    pts1=np.float32([[nw_x,nw_y],[ne_x,ne_y],[sw_x,sw_y],[se_x,se_y]])
    pts2=np.float32([[0,0],[wigth,0],[0,height],[wigth,height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (500,470))
    cv2.imshow('after',dst)
    key=cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()

    


# In[3]:


image_perspective('lenna.jpg',20,30,300,20,20,400,400,500)


# In[ ]:




