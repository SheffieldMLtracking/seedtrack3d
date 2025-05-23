import numpy as np
#np.set_printoptions(precision=3,suppress=True)
import cv2
import glob
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt



def convert_images(img,bg=255):
    a = int(1224*0.5)
    b = a+1224
    left = img.copy()
    left[:,a:b] = left[:,:1224]
    left[:,b:] = bg
    left[:,:a] = bg
    right = img.copy()
    right[:,a:b] = right[:,1224:]
    right[:,b:] = bg
    right[:,:a] = bg    
    imgs = [left[:,-1::-1],right]
    #imgs = [img[:,:1224][:,-1::-1],img[:,1224:]]
    return imgs
    
#def convert_images(img):
#    
#    left = img.copy()
#    left[:,1224:] = 0
#    right = img.copy()
#    right[:,:1224] = 0
#    imgs = [left,right[:,-1::-1]]
#    #imgs = [img[:,:1224],img[:,1224:][:,-1::-1]]
#    return imgs
    
def load_images(fname):
    img = cv2.imread(fname)
    greyscaleimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#.astype(float)
    return convert_images(greyscaleimg)
    

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
 
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objpoints = [[],[]]
        imgpoints = [[],[]]
        self.test_imgpoints = [[],[]]
        imgshape = [None,None]
        print("Calibration: %d images available" % len(image_fns))
        
        self.test_images = [] #provides debug images for testing later
        for imgindex,fname in enumerate(image_fns):
            objp = []
            for i in range(11):
                for j in range(5):
                    objp.append([j*2+(i%2),i,0])
                    #objp.append([(i+((j+1)%2)/2),j/2,0])
            objp = np.array(objp).astype(np.float32)*10.606601718
            self.objp = objp
            imgs = load_images(fname)
            if imgindex<5: self.test_images.append([])
            for cami,img in enumerate(imgs):
                isFound, circle_locations = cv2.findCirclesGrid(img, (5,11), flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
                if imgindex<5:#len(image_fns):
                    self.test_imgpoints[cami].append(circle_locations)
                    self.test_images[-1].append(img)
                else:
                    objpoints[cami].append(objp.copy())
                    imgpoints[cami].append(circle_locations)
                    imgshape[cami] = img.shape

        self.imgpoints = imgpoints        
        
        mtxs = []
        dists = []
        print("2d calibration...")
        for i in range(2):
            print("CAMERA %d" % i)
            #returnvalue, camera matrix, distortion coefficients, rotation and translation vectors 
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints[i], imgpoints[i], imgshape[i], None, None)
            mtxs.append(mtx)
            dists.append(dist)
            print('rmse:', ret)
        
        
        self.mtxs = mtxs
        self.dists = dists
        
        #build imagepoints list of list of numpy arrays in same shape but undistorted.
        undistorted_imgpoints = []
        for i in range(2):
            undist_imgps = []
            for imgps in imgpoints[i]:
                undist_imgps.append(cv2.undistortPoints(imgps,mtxs[i],dists[i],None,mtxs[i]))
            undistorted_imgpoints.append(undist_imgps)
        #undistorted_imgpoints = imgpoints
           
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)


        #stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
        #stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize[, R[, T[, E[, F[, flags[, criteria]]]]]]) -> retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F
        ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints[0], undistorted_imgpoints[0], undistorted_imgpoints[1], mtxs[0], dists[0], mtxs[1], dists[1], imgshape[0], criteria = criteria, flags=flags)
        
        self.R = R
        self.T = T
        self.E = E
        self.F = F
        self.newdists = [dist1,dist2]
        self.dists = [dist1,dist2] #should we use these or old ones?
        
        RTs = []
        RTs.append(np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1))
        RTs.append(np.concatenate([R, T], axis = -1))
        self.Ps = []
        for mtx,RT in zip(mtxs,RTs):
            self.Ps.append(mtx @ RT)
            

    def get_coordinates(self,imgpoints1,imgpoints2,undistort=True):
        
        #projMat1 = self.mtxs[0] @ cv2.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
        #projMat2 = self.mtxs[1] @ cv2.hconcat([R, T]) # R, T from stereoCalibrate


        if np.isscalar(imgpoints1[0]):
            if undistort:
                undistorted_imgpoints1 = cv2.undistortPoints(np.array(imgpoints1).astype(np.float32),self.mtxs[0],self.dists[0],None,self.mtxs[0])[0,0,:]
                undistorted_imgpoints2 = cv2.undistortPoints(np.array(imgpoints2).astype(np.float32),self.mtxs[1],self.dists[1],None,self.mtxs[1])[0,0,:] 
            else:
                undistorted_imgpoints1 = np.array(imgpoints1).astype(np.float32)
                undistorted_imgpoints2 = np.array(imgpoints2).astype(np.float32)
            p4d = cv2.triangulatePoints(self.Ps[0],self.Ps[1], undistorted_imgpoints1,undistorted_imgpoints2)
            p = (p4d[:3, :]/p4d[3, :])
            return p[:,0], p4d
            #return self.DLT(undistorted_imgpoints1,undistorted_imgpoints2)
        
        coords = []
        p4ds = []
        for p1,p2 in zip(imgpoints1,imgpoints2):
            if undistort:
                undistorted_p1 = cv2.undistortPoints(np.array(p1).astype(np.float32),self.mtxs[0],self.dists[0],None,self.mtxs[0])[0,0,:]
                undistorted_p2 = cv2.undistortPoints(np.array(p2).astype(np.float32),self.mtxs[1],self.dists[1],None,self.mtxs[1])[0,0,:]
            else:
                undistorted_p1 = p1
                undistorted_p2 = p2
            #coords.append(self.DLT(undistorted_p1,undistorted_p2))
            p4d = cv2.triangulatePoints(self.Ps[0],self.Ps[1], undistorted_p1,undistorted_p2)
            p = (p4d[:3, :]/p4d[3, :])
            coords.append(p[:,0])
            p4ds.append(p4d)
        return np.array(coords), np.array(p4ds)

    def check_tests(self,undistort=True):
        """
        Some calibration images were left out, this checks if the dots are correct distance apart
        Returns mean percentage error, in percent.
        """
        checks =[]
        for test in zip(self.test_imgpoints[0],self.test_imgpoints[1]):
            world = []
            
            world,_ = self.get_coordinates(test[0][:,0,:],test[1][:,0,:],undistort=undistort)
            v = world[0,:]-world[1+np.argmin(np.sum((world[0,:]-world[1:,:])**2,1)),:]
            distances= np.diag(np.sqrt(np.sum(((world[:,:,None]-world[:,:,None].T)**2),1)),1)
            avgdist = np.mean(np.array([distances[i:(i+4)] for i in range(0,len(distances),5)]))
            checks.append(avgdist)
        targetdistance = np.sqrt(2*15**2)
        averageobserveddistance = np.array(checks)
        return 100*np.mean(np.abs(averageobserveddistance-targetdistance))/targetdistance
       
class Seed_Trajectory:
    def __init__(self,cam_alignment, pathtodata, save_debug_images = None, smooth=2, threshold=7, undistort=True):
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
        self.trajectory = self.get_trajectory(pathtodata,undistort=undistort)
        
        
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
                #diff_imgs = [diff_greyscaleimg[:,:1224],diff_greyscaleimg[:,1224:]]#[:,-1::-1]]
                diff_imgs = convert_images(diff_greyscaleimg,bg=0)              
                coords = []
                
                #TEMP experiment undistorting image first...
                #for i in range(2):
                #    img = diff_imgs[i]
                #    diff_imgs[i] = cv2.undistort(img,self.cam_alignment.mtxs[i],self.cam_alignment.dists[i])
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

    def get_trajectory(self,pathtodata,undistort=True):
        image_fns = sorted(glob.glob(os.path.join(pathtodata,'*.tiff')))
        results = self.get_seed_2d_coordinates(image_fns)
        self.trajectories2d = results
        seedpath = []
        for r in results:
            if (r[1]<self.threshold) or (r[2]<self.threshold):
                continue
            seedpath.append([r[0]]+self.cam_alignment.get_coordinates(r[3],r[4],undistort=undistort)[0].tolist())
        return seedpath
