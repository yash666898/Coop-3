# core.py
import numpy as np
import cv2
from PIL import Image
import shutil

# Character density map (shared)
CHAR_DENSITY = {
    ' ': 0.0, '.': 0.05, ',': 0.1, ':': 0.15, ';': 0.2,
    '░': 0.25, '▒': 0.5, '▓': 0.75, '█': 1.0,
    '-': 0.4, '_': 0.45, '|': 0.5, '/': 0.5, '\\': 0.5,
    '#': 0.7, '@': 0.8, '%': 0.6
}

def get_terminal_size():
    size = shutil.get_terminal_size((80, 24))
    return size.columns, size.lines

def load_and_resize_image(path, aspect_ratio=2, target_w=None, target_h=None):
    img = Image.open(path).convert('RGB')
    term_w, term_h = get_terminal_size()
    new_w = term_w if target_w is None else target_w
    new_h = int(term_h * aspect_ratio) if target_h is None else target_h
    img = img.resize((new_w, new_h))
    return np.array(img)

def luminance_map(img):
    return 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]

def edge_map(lum):
    sobelx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)
    return magnitude, orientation

def classify_region(mag, lum, x, y, bg_thresh=20, shade_thresh=50, var_thresh=50):
    local_var = np.var(lum[max(0,y-1):y+2, max(0,x-1):x+2])
    if mag[y,x] < bg_thresh and local_var < var_thresh:
        return 'background'
    elif mag[y,x] < shade_thresh:
        return 'shading'
    else:
        return 'foreground'

def best_char_for_luminance(lum_val):
    target_density = lum_val / 255.0
    return min(CHAR_DENSITY.keys(),
               key=lambda c: abs(CHAR_DENSITY[c] - target_density))

def orientation_char(orientation):
    if -0.25 < orientation < 0.25:
        return '-'
    elif 1.2 < orientation < 1.8 or -1.8 < orientation < -1.2:
        return '|'
    elif orientation >= 0.25 and orientation < 1.2:
        return '/'
    elif orientation <= -0.25 and orientation > -1.2:
        return '\\'
    else:
        return '#'

def select_char(region, lum_val, orientation):
    if region in ['background', 'shading']:
        return best_char_for_luminance(lum_val)
    else:
        return orientation_char(orientation)

def floyd_steinberg_dither(lum):
    h, w = lum.shape
    lum = lum.copy().astype(float)
    for y in range(h-1):
        for x in range(1, w-1):
            old_pixel = lum[y, x]
            new_pixel = round(old_pixel / 51) * 51
            lum[y, x] = new_pixel
            error = old_pixel - new_pixel
            lum[y, x+1] += error * 7/16
            lum[y+1, x-1] += error * 3/16
            lum[y+1, x] += error * 5/16
            lum[y+1, x+1] += error * 1/16
    return lum

def ansi_color(r,g,b,text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
