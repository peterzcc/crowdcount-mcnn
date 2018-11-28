import numpy as np
import cv2
import os
import random
import pandas as pd
from mcnn.exr_utils import load_exr
class ExrImageDataLoader():
    def __init__(self, data_path, gt_path,mask_path = None, shuffle=False, gt_downsample=False, pre_load=False):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = self.get_data_path_files(data_path, fmt=".jpg")
        self.data_files.sort(key=lambda x: x[1])
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}        
        self.id_list = list(range(0,self.num_samples))
        self.masks = None
        if mask_path is not None:
            self.masks = {}
            for msk_fname in os.listdir(mask_path):
                if not msk_fname.endswith('.png'):
                    print("ignoring mask: {}".format(msk_fname))
                    continue
                msk = (cv2.imread(os.path.join(mask_path, msk_fname),0)>0).astype(np.uint8)
                ht, wd = msk.shape[0:2]
                msk = self.resize_img_4(msk)
                ht_1, wd_1 = msk.shape[0:2]

                if self.gt_downsample:
                    wd_1 = int(wd_1 / 4)
                    ht_1 = int(ht_1 / 4)
                    msk_small = cv2.resize(msk, (wd_1, ht_1))
                    self.masks[msk_fname[0:6]] = msk_small
                else:
                    self.masks[msk_fname[0:6]] = msk

        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for img_path, fname in self.data_files:
                assert fname[0:6] in self.masks
                img, den = self.process_img(img_path,fname)
                if img is None: continue
                blob = {'data': img,
                        'gt_density': den,
                        'fname': fname,
                        'mask': self.masks[fname[0:6]]}
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 100 == 0:                    
                    print('Loaded ', idx, '/', self.num_samples, 'files')
            self.id_list = list(range(0, idx))
            print('Completed Loading ', idx, 'files')

    def resize_img_4(self,img):
        ht = img.shape[0]
        wd = img.shape[1]
        ht_1 = int((ht / 4) * 4)
        wd_1 = int((wd / 4) * 4)
        return cv2.resize(img, (wd_1, ht_1))

    def get_dmap_path(self, img_fname):
        dmap_fname = os.path.splitext(img_fname)[0]+".exr"
        return os.path.join(self.gt_path, dmap_fname)

    def get_data_path_files(self, data_dir, fmt=".jpg"):
        data_files = []
        missing_files = []
        for path, subdirs, files in os.walk(data_dir):
            for fname in files:
                if os.path.isfile(os.path.join(path,fname)) \
                        and fname.endswith(fmt) \
                        and os.path.isfile(self.get_dmap_path(fname)):
                    data_files.append((path,fname))
                else:
                    missing_files.append(fname)
        print("missing:\n{}".format(missing_files))
        return data_files

    def process_img(self,img_path, fname):
        img = cv2.imread(os.path.join(img_path, fname), 0).astype(np.float32, copy=False)
        ht, wd = img.shape[0:2]
        img = self.resize_img_4(img)
        ht_1, wd_1 = img.shape[0:2]
        img = img.reshape((1, 1, img.shape[0], img.shape[1]))
        if self.gt_path is not None:
            den_fname = self.get_dmap_path(fname)
            den = load_exr(den_fname).astype(np.float32, copy=False)
        else:
            den = np.zeros(img.shape, dtype=np.float32)
        if self.gt_downsample:
            wd_1 = int(wd_1 / 4)
            ht_1 = int(ht_1 / 4)
        true_count = den.sum()
        den = cv2.resize(den, (wd_1, ht_1))
        current_count = den.sum()
        den = den * true_count/current_count if current_count > 0 else den
        den = den.reshape((1, 1, den.shape[0], den.shape[1]))
        if np.isnan(den).any():
            print("invalid dmap: {}".format(fname))
            return None, None
        return img, den

    def __iter__(self):
        if self.shuffle:            
            if self.pre_load:            
                random.shuffle(self.id_list)        
            else:
                random.shuffle(self.data_files)
        id_list = self.id_list
       
        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]    
                blob['idx'] = idx
            else:                    
                img_path, fname = self.data_files[idx]
                img, den = self.process_img(img_path,fname)
                if img is None: continue
                blob = {'data': img,
                        'gt_density': den,
                        'fname': fname,
                        'mask': self.masks[fname[0:6]]}
                
            yield blob
            
    def get_num_samples(self):
        return self.num_samples
                
        
            

