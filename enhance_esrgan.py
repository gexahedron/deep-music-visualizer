import numpy as np
import cv2
from os import listdir
from os.path import join
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import ESRGAN.RRDBNet_arch as arch


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--saturation_enhancement", type=float, default=1.3)
parser.add_argument("--upscale", dest='upscale', action='store_true')
parser.set_defaults(upscale=False)
args = parser.parse_args()

folder = args.input
new_folder = args.output
Path(new_folder).mkdir(parents=True, exist_ok=True)

# ['models/RRDB_ESRGAN_x4.pth','models/RRDB_PSNR_x4.pth','models/PPON_G.pth','models/PPON_D.pth']  
model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'
device = torch.device('cuda') 

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

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
        im_mod = im_mod * 1.0 / 255
        im_mod = torch.from_numpy(np.transpose(im_mod[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = im_mod.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()  
    else:
        output = im_mod  
    cv2.imwrite(join(new_folder, '%s.png' % im_name), output)
