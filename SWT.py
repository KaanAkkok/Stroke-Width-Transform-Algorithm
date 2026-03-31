import cv2
import numpy as np
from PIL import Image

def img_to_array(image):
    if isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, str):
        return cv2.imread(image)
    else:
        raise TypeError("Image must be a numpy array or a PIL Image")

def stroke_width_transform(image):
    try:
        image_array = img_to_array(image)
    except TypeError as e:
        print(e)
        return None
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    edges = cv2.Canny(gray, 50, 150)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    direction = np.arctan2(grad_y, grad_x)
    swt = np.full(gray.shape, np.inf)
    height, width = gray.shape
    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:
                dx = np.cos(direction[y, x])
                dy = np.sin(direction[y, x])
                cur_x, cur_y = x + 0.5, y + 0.5
                ray_len = 0
                while True:
                    cur_x += dx
                    cur_y += dy
                    ray_len += 1
                    ix, iy = int(cur_x), int(cur_y)
                    if ix < 0 or iy < 0 or ix >= width or iy >= height:
                        break
                    if edges[iy, ix] > 0:
                        swt[y, x] = min(swt[y, x], ray_len)
                        break
    return swt

def clean_swt(swt):
    swt_clean = np.nan_to_num(swt, nan=0.0, posinf=255, neginf=0)
    swt_clipped = np.clip(swt_clean, 0, 255).astype(np.uint8)
    swt_resized = cv2.resize(swt_clipped, (swt.shape[1], swt.shape[0]))
    return swt_resized

def save_image(image, path):
    cv2.imwrite(path, image)

if __name__ == "__main__":
    swt = stroke_width_transform("test1_roi.png")
    swt_resized = clean_swt(swt)
    save_image(swt_resized, "swt.png")
    print("swt.png saved")
    save_image(swt, "swt_original.png")
    print("swt_original.png saved")
    