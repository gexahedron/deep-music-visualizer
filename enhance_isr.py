import numpy as np
import cv2
from os import listdir
from os.path import join
from pathlib import Path
from tqdm import tqdm
import argparse
from ISR.models import RDN


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--upscale", dest='upscale', action='store_true')
parser.add_argument("--saturation_enhancement", type=float, default=1.3)
parser.set_defaults(upscale=False)
args = parser.parse_args()

folder = args.input
new_folder = args.output
Path(new_folder).mkdir(parents=True, exist_ok=True)

rdn = RDN(weights='noise-cancel')
files = listdir(folder)
files.sort()
for i in tqdm(range(len(files))):
    im_name = files[i].split('.')[0]    

    im_orig = cv2.imread(join(folder, '%s.png' % im_name))
    
    im_mod = im_orig.copy()
    
    hsv = cv2.cvtColor(im_mod, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = s * args.saturation_enhancement
    s = np.clip(s, 0, 255)
    im_mod = cv2.merge((h, s.astype(np.uint8), v))
    im_mod = cv2.cvtColor(im_mod, cv2.COLOR_HSV2BGR)

    if args.upscale:   
        im_mod = rdn.predict(im_mod)
    
    cv2.imwrite(join(new_folder, '%s.png' % im_name), im_mod)
