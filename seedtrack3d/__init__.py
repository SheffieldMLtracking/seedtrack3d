import numpy as np
#np.set_printoptions(precision=3,suppress=True)
import cv2
import glob
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


class Camera_Alignment:
    def DLT(self, point1, point2):
        P1 = self.Ps[0]
        P2 = self.Ps[1]
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
        return Vh[3,0:3]/Vh[3,3]
    
        
    def __init__(self,pathtodata):
        """
        pathtodata: should be path to folder containing calibration images, e.g. "~/seeds/250305/calibration"
        """
        image_fns = sorted(glob.glob(os.path.join(pathtodata,'*.tiff')))
 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = [[],[]]
        imgpoints = [[],[]]
        self.test_imgpoints = [[],[]]
        imgshape = [None,None]
        print("Calibration: %d images available" % len(image_fns))
        
        for imgindex,fname in enumerate(image_fns):
            objp = []
            for i in range(11):
                for j in range(5):
                    objp.append([j+(i%2)/2,i,0])
                    #objp.append([(i+((j+1)%2)/2),j/2,0])
            objp = np.array(objp).astype(np.float32)
            img = cv2.imread(fname)
            greyscaleimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs = [greyscaleimg[:,:1224],greyscaleimg[:,1224:][:,-1::-1]]
            for cami,img in enumerate(imgs):
                isFound, circle_locations = cv2.findCirclesGrid(img, (5,11), flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
                if imgindex<5:#len(image_fns):
                    self.test_imgpoints[cami].append(circle_locations)
                else:
                    objpoints[cami].append(objp.copy())
                    imgpoints[cami].append(circle_locations)
                    imgshape[cami] = img.shape
            #if imgindex==40:
            #    plt.figure()
            #    plt.imshow(img,cmap='gray')
            #    plt.plot(circle_locations[:,0,0],circle_locations[:,0,1],'+')
                
        
        mtxs = []
        dists = []
        print("2d calibration...")
        for i in range(2):
            print("CAMERA %d" % i)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[i], imgpoints[i], imgshape[i], None, None)
            mtxs.append(mtx)
            dists.append(dist)
            print('rmse:', ret)
        
        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints[0], imgpoints[0], imgpoints[1], mtxs[0], dists[0], mtxs[1], dists[1], imgshape[0], criteria = criteria, flags = stereocalibration_flags)
        
        RTs = []
        RTs.append(np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1))
        RTs.append(np.concatenate([R, T], axis = -1))
        self.Ps = []
        for mtx,RT in zip(mtxs,RTs):
            self.Ps.append(mtx @ RT)

    def get_coordinates(self,imgpoints1,imgpoints2):
        if np.isscalar(imgpoints1[0]): return self.DLT(imgpoints1,imgpoints2)
        
        coords = []
        for p1,p2 in zip(imgpoints1,imgpoints2):
            coords.append(self.DLT(p1,p2))
        return np.array(coords)

    def check_tests(self):
        """
        Some calibration images were left out, this checks if the dots are correct distance apart
        Returns mean percentage error, in percent.
        """
        checks =[]
        for test in zip(self.test_imgpoints[0],self.test_imgpoints[1]):
            world = []
            
            world = self.get_coordinates(test[0][:,0,:],test[1][:,0,:])
            v = world[0,:]-world[1+np.argmin(np.sum((world[0,:]-world[1:,:])**2,1)),:]
            distances= np.diag(np.sqrt(np.sum(((world[:,:,None]-world[:,:,None].T)**2),1)),1)
            avgdist = np.mean(np.array([distances[i:(i+4)] for i in range(0,len(distances),5)]))
            checks.append(avgdist)
        
        return 100*np.mean(np.abs(np.array(checks)-1))
       
class Seed_Trajectory:
    def __init__(self,cam_alignment, pathtodata, save_debug_images = None, smooth=2, threshold=7):
        """
        Compute a 3d path of a seed, given the images in 'pathtodata' and the camera alignment object in cam_alignment.

        Parameters:
         cam_alignment: An alignment object
         pathtodata: The path where the set of .tiff files reside.
         save_debug_images: Save a debug pdf, with every nth image saved; set to None to not output debug image.
         smooth: whether to smooth the images slightly (Gaussian filter), default = 2pixels.
         threshold: in the image differences what is a threshold for the seed's brightness vs the background noise.
        """
        self.save_debug_images = save_debug_images
        self.smooth = smooth
        self.pathtodata = pathtodata
        self.threshold = threshold
        self.cam_alignment = cam_alignment
        self.trajectory = self.get_trajectory(pathtodata)
        
        
    def get_seed_2d_coordinates(self,image_fns):
        """
        
        """
        if self.save_debug_images is not None:
            plt.figure(figsize=[5,2*len(image_fns)/self.save_debug_images])
        lastimg = None
        results = []
        for imgindex,fname in enumerate(image_fns):
            img_indexid = int(os.path.split(fname)[-1][-9:-5])
            img = cv2.imread(fname)
            greyscaleimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            if self.smooth is not None:
                greyscaleimg = ndimage.gaussian_filter(greyscaleimg, sigma=(self.smooth, self.smooth), order=0)
            
            if lastimg is not None:
                diff_greyscaleimg = greyscaleimg - lastimg
                diff_imgs = [diff_greyscaleimg[:,:1224],diff_greyscaleimg[:,1224:][:,-1::-1]]
                coords = []
                for diff_img in diff_imgs:
                    coords.append(np.unravel_index(diff_img.argmax(), diff_img.shape))
                #print(str(np.max(diff_imgs[0]))+' '+str(np.max(diff_imgs[1]))+str(coords))
                results.append([img_indexid,np.max(diff_imgs[0]),np.max(diff_imgs[1]),coords[0],coords[1]])
                if self.save_debug_images is not None:
                    if imgindex%self.save_debug_images==0:
                        for imgpairi in range(2):
                            plt.subplot(int(1+len(image_fns)/self.save_debug_images),2,int(2*(imgindex/self.save_debug_images)+imgpairi+1))
                            plt.imshow(diff_imgs[imgpairi][max(coords[imgpairi][0]-100,0):min(coords[imgpairi][0]+100,2047),max(coords[imgpairi][1]-100,0):min(coords[imgpairi][1]+100,1223)])
                            plt.clim([0,np.max(diff_imgs[imgpairi])])
                            plt.vlines([100,100],[25,125],[75,175],'w')
                            plt.hlines([100,100],[25,125],[75,175],'w')
                            plt.xticks([])
                            plt.yticks([])
                            if imgpairi==0: plt.title("%0.1f %0.1f " % (np.max(diff_imgs[0]),np.max(diff_imgs[1]))+str(coords))
            lastimg = greyscaleimg
        if self.save_debug_images:
            path_to_data, _ = os.path.split(image_fns[0])
            plt.savefig(os.path.join(path_to_data,'debug.pdf'))
        return results

    def get_trajectory(self,pathtodata):
        image_fns = sorted(glob.glob(os.path.join(pathtodata,'*.tiff')))
        results = self.get_seed_2d_coordinates(image_fns)
        
        seedpath = []
        for r in results:
            if (r[1]<self.threshold) or (r[2]<self.threshold):
                continue
            seedpath.append([r[0]]+self.cam_alignment.get_coordinates(r[3],r[4]).tolist())
        return seedpath
