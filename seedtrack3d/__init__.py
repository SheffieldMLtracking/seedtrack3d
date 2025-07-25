import numpy as np
import glob
import cv2
import scipy.ndimage as ndimage
import alignment
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def convert_images(img,bg=255):
    a = int(1224*0.5)
    b = a+1224
    left = img.copy().astype(float)
    left[:,a:b] = left[:,:1224]
    left[:,b:] = bg
    left[:,:a] = bg
    right = img.copy().astype(float)
    right[:,a:b] = right[:,1224:]
    right[:,b:] = bg
    right[:,:a] = bg    
    imgs = [left[:,-1::-1],right]
    return imgs
    
def load_images(fname):
    img = cv2.imread(fname)
    greyscaleimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return convert_images(greyscaleimg)

def get_seed_2d_coordinates(image_fns,save_debug_images=None,smooth=2):
    """
    image_fns = list of filenames to images of seeds.
    save_debug_images = whether to save debug images.
    smooth = amount to smooth the images (to help reduce noise), in pixels.
    """
    if save_debug_images is not None:
        plt.figure(figsize=[5,2*len(image_fns)/save_debug_images])
    lastimg = None
    results = []
    for imgindex,fname in enumerate(image_fns):
        print("%d/%d\r" % (imgindex,len(image_fns)),end="")
        img_indexid = int(os.path.split(fname)[-1][-9:-5])
        img = cv2.imread(fname)
        greyscaleimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        if smooth is not None:
            greyscaleimg = ndimage.gaussian_filter(greyscaleimg, sigma=(smooth, smooth), order=0)
        
        if lastimg is not None:
            diff_greyscaleimg = greyscaleimg - lastimg
            diff_imgs = convert_images(diff_greyscaleimg,bg=0)              
            coords = []
            
            for diff_img in diff_imgs:
                coords.append(np.unravel_index(diff_img.argmax(), diff_img.shape))
            results.append([img_indexid,np.max(diff_imgs[0]),np.max(diff_imgs[1]),coords[0],coords[1]])
            if save_debug_images is not None:
                if imgindex%save_debug_images==0:
                    for imgpairi in range(2):
                        plt.subplot(int(1+len(image_fns)/save_debug_images),2,int(2*(imgindex/save_debug_images)+imgpairi+1))
                        plt.imshow(diff_imgs[imgpairi][max(coords[imgpairi][0]-100,0):min(coords[imgpairi][0]+100,2047),max(coords[imgpairi][1]-100,0):min(coords[imgpairi][1]+100,2447)])
                        plt.clim([0,np.max(diff_imgs[imgpairi])])
                        plt.vlines([100,100],[25,125],[75,175],'w')
                        plt.hlines([100,100],[25,125],[75,175],'w')
                        plt.xticks([])
                        plt.yticks([])
                        if imgpairi==0: plt.title("%d: %0.1f %0.1f " % (imgindex,np.max(diff_imgs[0]),np.max(diff_imgs[1]))+str(coords))
        lastimg = greyscaleimg
    if save_debug_images:
        path_to_data, _ = os.path.split(image_fns[0])
        plt.savefig(os.path.join(path_to_data,'debug.pdf'))
    return results
