import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def apply_histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return final

def apply_retinex(img):
    img_float = img.astype(np.float32) + 1.0
    retinex = np.log10(img_float) - np.log10(cv2.GaussianBlur(img_float, (0, 0), sigmaX=30, sigmaY=30))
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    retinex = np.uint8(np.clip(retinex, 0, 255))
    return retinex

def apply_fusion_based_enhancement(img):
    contrast_enhanced = apply_clahe(img)
    detail_enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    fusion = cv2.addWeighted(contrast_enhanced, 0.5, detail_enhanced, 0.5, 0)
    return fusion

def apply_sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

img = cv2.imread('input.jpg')

clahe_img = apply_clahe(img)
hist_eq_img = apply_histogram_equalization(img)
retinex_img = apply_retinex(img)
fusion_img = apply_fusion_based_enhancement(img)
sharpened_img = apply_sharpening(img)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
plt.title('CLAHE')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(hist_eq_img, cv2.COLOR_BGR2RGB))
plt.title('Histogram Equalization')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(retinex_img, cv2.COLOR_BGR2RGB))
plt.title('Retinex-based Correction')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(fusion_img, cv2.COLOR_BGR2RGB))
plt.title('Fusion-based Enhancement')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB))
plt.title('Sharpened Image')

plt.tight_layout()
plt.show()
