import os
import pandas as pd
from skimage.transform import resize
import numpy as np
import imageio
import pydicom
from pydicom.pixel_data_handlers import util as dcm_util

import albumentations as A

import pickle

# https://albumentations-demo.herokuapp.com
aug_pipeline_aggressive = A.Compose([
    A.ShiftScaleRotate(),
    A.GridDistortion(p=0.5, num_steps=5),
])
# below 2 were removed from above after epoch 6
# A.Cutout(p=0.5, num_holes=8, max_h_size=8, max_w_size=8),
# A.ChannelShuffle(p=0.5),

aug_pipeline = A.Compose([
    A.ShiftScaleRotate(p=0.5),
])
MIN_VAL,MAX_VAL = -1000,1000

class DataLoader():
    def __init__(self, dataset_name, augment=False, img_res=(256,256)):
        
        if dataset_name == 'c4kc-kits':
            self.mydf = pd.read_csv("../prepare/data_kits.csv")

        elif dataset_name == 'circles':
            self.mydf = pd.read_csv("../prepare/data_circles.csv")

        elif dataset_name == 'case_10024':
            cols =['pre_contrast', 'corticomedullary']
            # cols = ['pre_contrast', 'registered']
            # open so pd df only has 2 columns: pre_contrast and corticomedullary
            self.mydf = pd.read_csv("../prepare/case_10024.csv", usecols=cols)
            # self.mydf = pd.read_csv("../prepare/data_small.csv", usecols=cols)

            self.pre_slices = []
            self.con_slices = []

            # print(self.mydf.head())

        else:
            raise NotImplementedError()

        self.dataset_name = dataset_name
        self.img_res = img_res
        self.augment = augment

    def load_data(self, domain, batch_size=1):
        
        if self.dataset_name == 'c4kc-kits':
            path = list(self.mydf[self.mydf.series_description==domain].dcm_file)

        elif self.dataset_name == 'circles':
            if domain == 'heterogeneous':
                path = list(self.mydf[self.mydf["patient-id"].str.contains("50.0")].dcm_file)

            else:
                path = list(self.mydf[~self.mydf["patient-id"].str.contains("50.0")].dcm_file)

        elif self.dataset_name == 'case_10024':
            if domain == 'pre_contrast':
                seri_list = list(self.mydf.pre_contrast)

                for seri_file in seri_list:
                    with open(seri_file, 'r') as f:
                        contents = f.read()
                    slices = contents.split("\n")
                    self.pre_slices.extend(slices)

                #  creates list of paths to each slice
                path = []
                slice_list = list(self.pre_slices)
                path.extend(filter(None, slice_list))
                
                # writes paths to text file (for debugging)
                # textfile = open('filtered_pre_slices_paths.txt', 'w')
                # for element in path:
                #     textfile.write(element + '\n')
                # textfile.close()

            else:
                seri_list = list(self.mydf.corticomedullary)

                for seri_file in seri_list:
                    with open(seri_file, 'r') as f:
                        contents = f.read()
                    slices = contents.split("\n")
                    self.con_slices.extend(slices)

                #  creates list of paths to each slice
                path = []
                slice_list = list(self.con_slices)
                path.extend(filter(None, slice_list))

                # writes paths to text file (for debugging)
                # textfile = open('filtered_con_slices_paths.txt', 'w')
                # for element in path:
                #     textfile.write(element + '\n')
                # textfile.close()

        else:
            raise NotImplementedError()

        # print(len(self.pre_slices), len(self.con_slices))
        # input("waiting...")
        print(domain, len(path))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            img = resize(img, self.img_res)
            if self.augment:
                augmented = aug_pipeline(
                    image=img,
                )
                img = augmented['image']
            imgs.append(img)
        
        imgs = np.array(imgs)
        #print(imgs.shape)
        return imgs

    def load_batch(self, batch_size=1):
        if self.dataset_name == 'c4kc-kits':
            path_A = list(self.mydf[self.mydf.series_description=="noncontrast"].dcm_file)
            path_B = list(self.mydf[self.mydf.series_description=="late"].dcm_file)

        elif self.dataset_name == 'circles':
            path_A = list(self.mydf[self.mydf["patient-id"].str.contains("50.0")].dcm_file)
            path_B = list(self.mydf[~self.mydf["patient-id"].str.contains("50.0")].dcm_file)
            
        elif self.dataset_name == 'case_10024':
            pre_seri_list = list(self.mydf.pre_contrast)

            for pre_seri_file in pre_seri_list:
                with open(pre_seri_file, 'r') as f:
                    contents = f.read()
                pre_slices = contents.split("\n")
                self.pre_slices.extend(pre_slices)

            #  creates list of paths to each slice
            path_A = []
            pre_slice_list = list(self.pre_slices)
            path_A.extend(filter(None, pre_slice_list))

            con_seri_list = list(self.mydf.corticomedullary)

            for con_seri_file in con_seri_list:
                with open(con_seri_file, 'r') as f:
                    contents = f.read()
                con_slices = contents.split("\n")
                self.con_slices.extend(con_slices)

            #  creates list of paths to each slice
            path_B = []
            con_slice_list = list(self.con_slices)
            path_B.extend(filter(None, con_slice_list))
            
        else:
            raise NotImplementedError()


        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size
        # print(len(path_A), len(path_B), self.n_batches, total_samples)
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):

                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = resize(img_A, self.img_res)
                img_B = resize(img_B, self.img_res)
                

                if self.augment:
                    augmented = aug_pipeline(
                        image=img_A,
                    )
                    img_A = augmented['image']

                    augmented = aug_pipeline(
                        image=img_B,
                    )
                    img_B = augmented['image']

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)
            imgs_B = np.array(imgs_B)
            #print(imgs_A.shape,imgs_B.shape)
            yield imgs_A, imgs_B

    def load_img(self, path):
        raise NotImplementedError()
        img = self.imread(path)
        img = resize(img, self.img_res)
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        ds = pydicom.dcmread(path, force=True)
        # ds = pydicom.dcmread(path)
        arr = dcm_util.apply_modality_lut(ds.pixel_array, ds)
        arr = arr.astype(np.float)
        arr = np.expand_dims(arr,axis=-1)   # converts 2D to 3D
        # shape 512,512,1
        arr = (((arr-MIN_VAL)/(MAX_VAL-MIN_VAL)).clip(0,1))*2-1.    # may need to rescale properly, goes to -1 to 1 now
        # min,max -1,1
        return arr

if __name__ == "__main__":
    
    os.makedirs('static',exist_ok=True)

    # dl = DataLoader('c4kc-kits',augment=True)
    
    # out = dl.load_data("noncontrast",batch_size=2)
    # assert(out.shape==(2,256,256,1))
    
    # out = dl.load_data("late",batch_size=2)
    # assert(out.shape==(2,256,256,1))
    # batch_size = 8
    # for n,batch in zip(range(1),dl.load_batch(batch_size=batch_size)):
    #     A, B = batch
    #     A = (255*(A+1)/2).astype(np.uint8)
    #     B = (255*(B+1)/2).astype(np.uint8)
    #     assert(A.shape==(batch_size,256,256,1))
    #     assert(B.shape==(batch_size,256,256,1))
    #     for n in range(batch_size):
    #         imageio.imwrite(f"static/noncontrast-{n}.png",A[n,:,:].squeeze())
    #         imageio.imwrite(f"static/late-{n}.png",B[n,:,:].squeeze())

    # dl = DataLoader('circles',augment=True)
    
    # out = dl.load_data("heterogeneous",batch_size=2)
    # print(out.shape)
    # assert(out.shape==(2,256,256,1))
    
    # out = dl.load_data("homogeneous",batch_size=2)
    # assert(out.shape==(2,256,256,1))
    # batch_size = 8
    # for n,batch in zip(range(1),dl.load_batch(batch_size=batch_size)):
    #     A, B = batch
    #     A = (255*(A+1)/2).astype(np.uint8)
    #     B = (255*(B+1)/2).astype(np.uint8)
    #     assert(A.shape==(batch_size,256,256,1))
    #     assert(B.shape==(batch_size,256,256,1))
    #     for n in range(batch_size):
    #         imageio.imwrite(f"static/heterogeneous-{n}.png",A[n,:,:].squeeze())
    #         imageio.imwrite(f"static/homogeneous-{n}.png",B[n,:,:].squeeze())

    dl = DataLoader('case_10024',augment=True)
    
    out = dl.load_data("pre_contrast",batch_size=2)
    print(out.shape)
    assert(out.shape==(2,256,256,1))
    
    out = dl.load_data("corticomedullary",batch_size=2)
    print(out.shape)
    assert(out.shape==(2,256,256,1))
    batch_size = 8
    for n,batch in zip(range(1),dl.load_batch(batch_size=batch_size)):
        A, B = batch
        A = (255*(A+1)/2).astype(np.uint8)
        B = (255*(B+1)/2).astype(np.uint8)
        assert(A.shape==(batch_size,256,256,1))
        assert(B.shape==(batch_size,256,256,1))
        for n in range(batch_size):
            imageio.imwrite(f"static/pre_contrast-{n}.png",A[n,:,:].squeeze())
            imageio.imwrite(f"static/corticomedullary-{n}.png",B[n,:,:].squeeze())

    print("done.")

'''
python data_loader.py
'''