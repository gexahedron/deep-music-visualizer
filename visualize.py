import cv2
import librosa
import argparse
import numpy as np
import random
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from os.path import join
from pytorch_pretrained_biggan import (BigGAN, truncated_noise_sample)

#get input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--song", required=True)
parser.add_argument("--folder", required=True, type=str)
parser.add_argument("--resolution", default='512')
parser.add_argument("--offset", type=float, default=0.0)
parser.add_argument("--duration", type=float, default=15)
parser.add_argument("--pitch_sensitivity", type=int, default=220)
parser.add_argument("--tempo_sensitivity", type=float, default=0.25)
parser.add_argument("--depth", type=float, default=1)
parser.add_argument("--classes", nargs='+', type=int)
parser.add_argument("--sort_classes_by_power", type=int, default=0)
parser.add_argument("--jitter", type=float, default=0.5)
parser.add_argument("--frame_length", type=int, default=512)
parser.add_argument("--truncation", type=float, default=1)
parser.add_argument("--smooth_factor", type=int, default=16)
parser.add_argument("--use_previous_classes", type=int, default=0)
parser.add_argument("--use_previous_vectors", type=int, default=0)
args = parser.parse_args()


#read song
if args.song:
    song = args.song
    print('\nReading audio \n')
    y, sr = librosa.load(song, offset=args.offset, duration=0.6*args.duration)
else:
    raise ValueError("you must enter an audio file name in the --song argument")

#set model name based on resolution
model_name='biggan-deep-' + args.resolution

frame_length=args.frame_length

#set pitch sensitivity
pitch_sensitivity=(300-args.pitch_sensitivity) * 512 / frame_length

#set tempo sensitivity
tempo_sensitivity=args.tempo_sensitivity * frame_length / 512

#set depth
depth=args.depth

#set number of classes
num_classes=len(args.classes)

#set sort_classes_by_power
sort_classes_by_power=args.sort_classes_by_power

#set jitter
jitter=args.jitter

#set truncation
truncation=args.truncation

#set batch size
batch_size=args.smooth_factor

#set use_previous_classes
use_previous_vectors=args.use_previous_vectors

#set use_previous_vectors
use_previous_classes=args.use_previous_classes


#set smooth factor
if args.smooth_factor > 1:
    smooth_factor=int(args.smooth_factor * 512 / frame_length)
else:
    smooth_factor=args.smooth_factor




#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################################
########################################
########################################
########################################
########################################


#create spectrogram
spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=frame_length)

#get mean power at each time point
specm=np.mean(spec,axis=0)

#compute power gradient across time points
gradm=np.gradient(specm)

#set max to 1
gradm = gradm/np.max(gradm)
#set negative gradient time points to zero
gradm = gradm.clip(min=0)

#normalize mean power between 0-1
specm = (specm-np.min(specm))/np.ptp(specm)

#create chromagram of pitches X time points
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=frame_length,
                                    n_chroma=len(args.classes), bins_per_octave=3*len(args.classes))

#sort pitches by overall power
chromasort = np.argsort(np.mean(chroma,axis=1))[::-1]



########################################
########################################
########################################
########################################
########################################


if args.classes:
    classes = args.classes
    if len(classes) not in [12, num_classes]:
        raise ValueError("The number of classes entered in the --class argument must equal 12 or [num_classes] if specified")
elif args.use_previous_classes == 1:
    cvs=np.load('class_vectors.npy')
    classes=list(np.where(cvs[0]>0)[0])
else: #select 12 random classes
    cls1000=list(range(1000))
    random.shuffle(cls1000)
    classes=cls1000[:12]


if sort_classes_by_power==1:
    classes=[classes[s] for s in np.argsort(chromasort[:num_classes])]


#initialize first class vector
cv1=np.zeros(1000)
for pi, p in enumerate(chromasort[:num_classes]):
    if num_classes < 12:
        cv1[classes[pi]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]
    else:
        cv1[classes[p]] = chroma[p][np.min([np.where(chrow>0)[0][0] for chrow in chroma])]

#initialize first noise vector
nv1 = truncated_noise_sample(truncation=truncation)[0]

# initialize list of class and noise vectors
class_vectors=[cv1]
noise_vectors=[nv1]

# initialize previous vectors (will be used to track the previous frame)
cvlast = cv1
nvlast = nv1


# initialize the direction of noise vector unit updates
update_dir = np.zeros(128)
for ni, n in enumerate(nv1):
    if n < 0:
        update_dir[ni] = 1
    else:
        update_dir[ni] = -1


# initialize noise unit update
update_last = np.zeros(128)


########################################
########################################
########################################
########################################
########################################


#get new jitters
def new_jitters(jitter):
    jitters = np.zeros(128)
    for j in range(128):
        if random.uniform(0, 1) < 0.5:
            jitters[j] = 1
        else:
            jitters[j] = 1 - jitter
    return jitters


#get new update directions
def new_update_dir(nv2, update_dir):
    for ni, n in enumerate(nv2):
        if n >= 2 * truncation - tempo_sensitivity:
            update_dir[ni] = -1

        elif n < -2 * truncation + tempo_sensitivity:
            update_dir[ni] = 1
    return update_dir


#smooth class vectors
def smooth(class_vectors, smooth_factor):
    class_vectors = class_vectors[:(len(class_vectors) // smooth_factor) * smooth_factor]
    for i in range(smooth_factor):
        class_vectors.append(class_vectors[i])

    if smooth_factor==1:
        return class_vectors

    class_vectors_terp = []
    for c in range(int(np.floor(len(class_vectors) / smooth_factor) - 1)):
        ci = c * smooth_factor
        # print('c:', int(ci), int(ci) + smooth_factor, int(ci) + smooth_factor * 2)
        cva = np.mean(class_vectors[int(ci):int(ci) + smooth_factor], axis=0)
        cvb = np.mean(class_vectors[int(ci) + smooth_factor:int(ci) + smooth_factor * 2], axis=0)
        for j in range(smooth_factor):
            cvc = cva * (1 - j / (smooth_factor - 1)) + cvb * (j / (smooth_factor - 1))
            class_vectors_terp.append(cvc)

    class_vectors_terp = class_vectors_terp[:-1]
    return np.array(class_vectors_terp)


#normalize class vector between 0-1
def normalize_cv(cv2):
    min_class_val = min(i for i in cv2 if i != 0)
    for ci,c in enumerate(cv2):
        if c==0:
            cv2[ci]=min_class_val
    cv2=(cv2-min_class_val)/np.ptp(cv2)

    return cv2


print('\nGenerating input vectors \n')

for i in tqdm(range(len(gradm))):

    #print progress
    pass

    #update jitter vector every 100 frames by setting ~half of noise vector units to lower sensitivity
    if i % 200 == 0:
        jitters = new_jitters(jitter)

    #get last noise vector
    nv1=nvlast

    #set noise vector update based on direction, sensitivity, jitter, and combination of overall power and gradient of power
    update = np.array([tempo_sensitivity for k in range(128)]) * (gradm[i]+specm[i]) * update_dir * jitters

    #smooth the update with the previous update (to avoid overly sharp frame transitions)
    update=(update+update_last*3)/4

    #set last update
    update_last=update

    #update noise vector
    nv2=nv1+update

    #append to noise vectors
    noise_vectors.append(nv2)

    #set last noise vector
    nvlast=nv2

    #update the direction of noise units
    update_dir=new_update_dir(nv2, update_dir)

    #get last class vector
    cv1=cvlast

    #generate new class vector
    cv2=np.zeros(1000)
    for j in range(num_classes):

        cv2[classes[j]] = (cvlast[classes[j]] + ((chroma[chromasort[j]][i])/(pitch_sensitivity)))/(1+(1/((pitch_sensitivity))))

    #if more than 6 classes, normalize new class vector between 0 and 1, else simply set max class val to 1
    if num_classes > 6:
        cv2=normalize_cv(cv2)
    else:
        cv2=cv2/np.max(cv2)

    #adjust depth
    cv2=cv2*depth

    #this prevents rare bugs where all classes are the same value
    if np.std(cv2[np.where(cv2!=0)]) < 0.0000001:
        cv2[classes[0]]=cv2[classes[0]]+0.01

    #append new class vector
    class_vectors.append(cv2)

    #set last class vector
    cvlast=cv2


#interpolate between class vectors of bin size [smooth_factor] to smooth frames
class_vectors = smooth(class_vectors, smooth_factor)
noise_vectors = smooth(noise_vectors, smooth_factor)


#check whether to use vectors from last run
if use_previous_vectors==1:
    #load vectors from previous run
    class_vectors=np.load('class_vectors.npy')
    noise_vectors=np.load('noise_vectors.npy')
else:
    #save record of vectors for current video
    np.save('class_vectors.npy',class_vectors)
    np.save('noise_vectors.npy',noise_vectors)



########################################
########################################
########################################
########################################
########################################


#convert to Tensor
noise_vectors = torch.Tensor(np.array(noise_vectors))
class_vectors = torch.Tensor(np.array(class_vectors))


#Generate frames in batches of batch_size

print('\n\nGenerating frames \n')

# Load pre-trained model
model = BigGAN.from_pretrained(model_name)
model = model.to(device)

#send to CUDA if running on GPU
noise_vectors=noise_vectors.to(device)
class_vectors=class_vectors.to(device)



Path(args.folder).mkdir(parents=True, exist_ok=True)

idx = 0
for i in tqdm(range(len(class_vectors) // batch_size + 1)):
    if i*batch_size > len(class_vectors):
        torch.cuda.empty_cache()
        break

    #get batch
    noise_vector=noise_vectors[i*batch_size:(i+1)*batch_size]
    class_vector=class_vectors[i*batch_size:(i+1)*batch_size]

    # Generate images
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    output_cpu=output.cpu().data.numpy()

    #convert to image array and add to frames
    for out in output_cpu:
        idx += 1
        im = np.array(out)
        im = (np.moveaxis(im, 0, -1) + 1) / 2
        im = (im * 255).astype(np.uint8)
        im_pil = Image.fromarray(im)
        im_pil.save(join(args.folder, '%03d.png' % idx))

    #empty cuda cache
    torch.cuda.empty_cache()
