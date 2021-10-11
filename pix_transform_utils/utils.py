import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import numpy as np
import scipy.misc

def downsample(image, scaling_factor):
    image = F.avg_pool2d(torch.from_numpy(image).unsqueeze(0).double(), scaling_factor)
    image = image.squeeze().numpy()
    return image


def upsample(image, scaling_factor, orig_guide_res=None):

    lrh, lrw = image.shape

    # Upsample regularly
    B = TVF.resize(image.unsqueeze(0), (lrh*scaling_factor,lrw*scaling_factor) ).squeeze()


    if orig_guide_res is not None:
        # handle the boarder cases, where the dimension were odd in the downsampling process
        outimg = torch.zeros(orig_guide_res, dtype=B.dtype)

        outimg[:B.shape[0], :B.shape[1]] = B 
        hd = orig_guide_res[0] - B.shape[0] 
        wd = orig_guide_res[1] - B.shape[1] 

        if hd > 0:
            outimg[ B.shape[0]:, :] = outimg[ B.shape[0]-1:B.shape[0],:].tile(hd,1)
        
        if wd > 0:
            outimg[ :, B.shape[1]:] = outimg[ :, B.shape[1]-1:B.shape[1]].tile(1,wd)

        B = outimg

    return B


def align_images(img_s,img_t,limits=(-1.,1.),steps=25):

    image_size = img_t.shape[0]

    maximum_limit = int(np.ceil(np.max(np.abs(np.array(limits)))))
    mask = np.zeros_like(img_s)
    mask[maximum_limit:-maximum_limit,maximum_limit:-maximum_limit] = 1.

    x_or_y = np.array(list(range(0, int(image_size)))).astype(float)
    img_t_shifter = scipy.interpolate.RectBivariateSpline(x_or_y, x_or_y, img_t)

    delta = np.linspace(limits[0],limits[1],steps)
    mse_best = 1e9
    x_best = 0
    y_best = 0
    for i in range(0,steps):
        for j in range(0,steps):

            x_grid, y_grid = np.meshgrid(x_or_y + delta[i], x_or_y + delta[j], indexing="ij")
            img_t_shifted = img_t_shifter.ev(x_grid, y_grid)

            mse = np.mean((mask*(img_t_shifted-img_s))**2)
            if mse < mse_best:
                mse_best = mse
                x_best = delta[i]
                y_best = delta[j]

    x_grid, y_grid = np.meshgrid(x_or_y + x_best, x_or_y + y_best, indexing="ij")
    img_t_shifted = img_t_shifter.ev(x_grid, y_grid)

    img_t_shifted = img_t_shifted[maximum_limit:-maximum_limit,maximum_limit:-maximum_limit]
    img_s = img_s[maximum_limit:-maximum_limit,maximum_limit:-maximum_limit]

    return img_s,img_t_shifted
