import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from pathlib import Path
from tqdm import tqdm
import argparse
from ISR.models import RDN


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
args = parser.parse_args()

folder = args.input
new_folder = args.output
Path(new_folder).mkdir(parents=True, exist_ok=True)

rdn = RDN(weights='noise-cancel')
files = os.listdir(folder)
files.sort()
for i in tqdm(range(len(files))):
    im_name = files[i].split('.')[0]    

    print(im_name)
    im_orig = cv2.imread(join(folder, '%s.png' % im_name))
    
    im_mod = im_orig.copy()
    
#     im_mod = cv2.pyrMeanShiftFiltering(im_mod, 5, 5)

#     lab = cv2.cvtColor(im_mod, cv2.COLOR_BGR2LAB)
#     lighting, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
#     lighting = clahe.apply(lighting)
#     im_mod = cv2.merge((lighting.astype(np.uint8), a, b))
#     im_mod = cv2.cvtColor(im_mod, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(im_mod, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s * 1.3
    s = np.clip(s, 0, 255)
    im_mod = cv2.merge((h, s.astype(np.uint8), v))
    im_mod = cv2.cvtColor(im_mod, cv2.COLOR_HSV2BGR)
    
#     im_mod = cv2.resize(im_mod, (2048, 2048), interpolation=cv2.INTER_LANCZOS4)
    im_mod = rdn.predict(im_mod)
#     im_mod = cv2.pyrMeanShiftFiltering(im_mod, 10, 10)
    
    cv2.imwrite(join(new_folder, '%s.png' % im_name), im_mod)
print('done!!!')
