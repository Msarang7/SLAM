import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

class Extractor(object):


    def __init__(self,k):
        self.orb = cv2.ORB_create(1)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.k = k
        self.kinv = np.linalg.inv(self.k)


    def denormalize(self,pt):
        ret =  np.dot(self.kinv,[pt[0],pt[1],1.0])
        ret /= ret[2]
        print(ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):

        # for ry in range(0, img.shape[0], img.shape[0]//self.gy):
        #     for rx in range(0, img.shape[1], img.shape[1]//self.gx):
        #         img_chunk = img[ry: ry+ self.gy, rx: rx+ self.gx]
        #         print(img_chunk.shape)
        #         kp, des = self.orb.detectAndCompute(img_chunk, None)

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3 )
        kps = [cv2.KeyPoint(x=f[0][0], y = f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'],k=2)
            for m,n in matches :
                if m.distance < 0.75*  n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1,kp2))


        if len(ret) > 0:
            ret = np.array(ret)
            # subtract to move to 0
            ret[:, :, 0] -= img.shape[0]//2
            ret[:, :, 1] -= img.shape[1]//2
            #print(ret.shape)

            # filter
            model, inliers = ransac((ret[:,0],ret[:,1]), FundamentalMatrixTransform,
                                    min_samples = 8, residual_threshold = 1, max_trials = 100)

            ret = ret[inliers]
            s,v,d = np.linalg.svd(model.params)
            print(v)

        self.last = {'kps':kps, 'des':des}
        return ret