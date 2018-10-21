#!/usr/bin/env python
# coding: utf-8

# # imports

# In[1]:


from __future__ import print_function
import numpy as np
import os, shutil
import sys
import cv2
import argparse
import time
#sys.path.append("..")  # Adds higher directory to python modules path.
#from img_to_vec import Img2Vec
#from PIL import Image


# # FaceDetector Class

# In[15]:


class FaceDetector:
    def __init__(self, detector = 'haar', standalone = False, output = './UPCfacesB/', samples = 50):
        self.detector = detector
        self.standalone = standalone
        self.video = cv2.VideoCapture('/dev/video0')
        #self.img2vec = Img2Vec(model='resnet-152')
        self.samples = samples
        self.pause = False
        self.wait = 15
        self.margin = 55
        self.cascPath = ''
        if self.detector == 'haar':
            self.cascPath="haarcascade_frontalface_alt2.xml"
        elif self.detector == 'lbp': 
            self.cascPath="lbpcascade_frontalface_improved.xml"
        else:
            exit()
        self.faceCascade=cv2.CascadeClassifier(self.cascPath)
        self.output = output
        self.new_dir = ''
        self.end = False
        
    def CreateDatasetDir(self):        
        try:
            # Create target Directory
            os.mkdir(self.output)
            print("Directory" , self.output ,  "Created ")
        except FileExistsError:
            print("Directory" , self.output ,  "already exists")

    def SetupDir(self):

        dirs = os.listdir(self.output)
        
        if len(dirs) > 0 and len(os.listdir(self.output + 'faces_' + str(len(dirs)) + '/') ) < self.samples:
            print('Directory is less than {} samples'.format(self.samples))
            self.new_dir = self.output + 'faces_' + str(len(dirs)) + '/'
            for the_file in os.listdir(self.new_dir):
                file_path = os.path.join(self.new_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            print('Directory' , self.new_dir, len(os.listdir(self.new_dir)) ,  'gets overwritten')
        else:    
            print('Directory is not empty')
            self.new_dir = self.output + 'faces_' + str(len(dirs)+1) + '/'
            try:
                os.mkdir(self.new_dir)
                print('Directory' , self.new_dir ,  'created')
            except FileExistsError as e:
                        print(e)
        
    def Detect(self):        
        k = 0
        frame_it=-1
        face_it=0
        ret, frame=self.video.read()
        mask = np.zeros(frame.shape[:],dtype=np.uint8)
        x0 = 0
        y0 = 0
        accum = 0
        index = []
        vec = None
        aux = 0
        while not self.end:
            
                
            ret, frame=self.video.read()
            if ret:
                frame_it+=1
                #gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces=self.faceCascade.detectMultiScale(frame,scaleFactor=1.01,minNeighbors=3,minSize=(140,140),flags=cv2.CASCADE_SCALE_IMAGE)
                if k == 0 and len(faces) > 0:
                    self.SetupDir()
                mask = np.zeros(frame.shape[:],dtype=np.uint8)
                for (x,y,w,h) in faces:
                    if (x0 == 0 and y0 == 0) or abs(x-x0)+abs(y-y0) < self.margin:
                        face_it+=1
                        mask=cv2.ellipse(mask,(x+w//2,y+int(h/2)),(int(w/3),int(h/2)),0,0,360,(255,255,255),-1)
                        mask=np.bitwise_and(frame,mask)
                        crop=mask[y:y+h,x:x+w]
                        #start = time.time()
                        #vec = self.img2vec.get_vec(Image.fromarray(crop))
                        #end = time.time()
                        #print(end - start, 'secs')
                        #print(vec.shape)
                        cv2.imwrite(self.new_dir + 'fr{}_f{}.jpg'.format(frame_it,face_it),crop)
                        
                        k += 1
#                         x0 = x
#                         y0 = y
#                         accum = 0
#                     elif abs(x-x0)+abs(y-y0) >= self.margin:
#                         accum += 1

                for (x,y,w,h) in faces:
                    if (x0 == 0 and y0 == 0) or abs(x-x0)+abs(y-y0) < self.margin:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        x0 = x
                        y0 = y
                        accum = 0
                    elif abs(x-x0)+abs(y-y0) >= self.margin:
                        accum += 1

                if accum > self.wait:
                    accum = 0
                    x0 = y0 = 0

                cv2.imshow('Video',frame)
                #if len(faces) > 0:
                    #print('Num Faces: {0}, ({1}, {2})'.format(len(faces), int(x0), int(y0)),end='\r')
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    self.end = True                    
                    break
                
                if c & 0xFF == ord('p'):
                    self.pause = True
                    print('Detection Paused')
                
                if (k >= self.samples):
                    print('Capture Success')
                    self.pause = True
                    
                if self.pause:
                    k = 0
                    face_it = 0
                    frame_it = -1
                    self.IdleVideo()
                    #self.Detect()
        self.video.release()
        cv2.destroyAllWindows()
        

        
    def IdleVideo(self):

        while self.pause:
            ret, frame = self.video.read()
            if ret:
                cv2.imshow('Video',frame)
                c = cv2.waitKey(1)
                if c & 0xFF == ord('q'):
                    self.end = True
                    break
                if c & 0xFF==ord('u'):
                    self.pause = False
                    print('Detection Resumed')
#                     elif 0xFF==ord('q'): 
#                         self.video.release()
#                         cv2.destroyAllWindows()
#                         break
        
                    
            
        
            
# def main():
#   parser = argparse.ArgumentParser()
#   parser.add_argument('foo')
#   parser.add_argument('bar')
#   args = parser.parse_args()
#   c1 = MyClass(args.foo, args.bar)
#   args_dict = vars(args)
#   c2 = MyClass(**args_dict)


# In[17]:


def main():
    fd = FaceDetector(samples=10)
    fd.CreateDatasetDir()
    fd.Detect()
    
main()


