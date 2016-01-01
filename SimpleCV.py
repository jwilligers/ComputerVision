
# coding: utf-8

# # Hello World

# In[ ]:

from SimpleCV import Camera,Color,Display,Image
camera = Camera()
disp = Display()
while disp.isNotDone():
    image = camera.getImage()
    image.save(disp)
print("Done")
exit()


# # Detect Yellow Object

# In[1]:

from SimpleCV import Camera,Color,Display,Image
camera = Camera()
disp = Display()
while disp.isNotDone():
    image = camera.getImage()
    yellow = image.colorDistance(Color.YELLOW).binarize(140).invert()
    onlyYellow = image-yellow
    onlyYellow.save(disp)
    
print("Done")
exit()


# # Face Detection

# In[ ]:

from SimpleCV import *
camera = Camera()
disp = Display()
segment = HaarCascade("face.xml")

while disp.isNotDone():
    image = camera.getImage()
    face = image
    autoface = image.findHaarFeatures(segment)
    if ( autoface is not None ):
        face = autoface[-1].crop()
    face.save(disp)
print("Done")
exit()

