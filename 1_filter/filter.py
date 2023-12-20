import timm
import pathlib
import numpy as np
import cv2
import os
import pandas as pd
import albumentations as A
import slide_tools
from torchvision import transforms as T
from tqdm import tqdm
from torch import nn
import json
import torch
import argparse

cv2.setNumThreads(1)

def main(args):
    ### Transform ###
    resize = A.Resize(224, 224, interpolation=cv2.INTER_CUBIC)
    def resize_wrapper(img):
        return resize(image=img)["image"]

        ### Data ###
    frame = pd.read_csv(args.csv)
    frame = frame.iloc[:10]

    rootify = lambda path: os.path.join(args.root, path)
    
    ds = slide_tools.tile_level.TileLevelDataset(
        slide_paths=frame.slide.apply(rootify),
        annotation_paths=frame.annotation.apply(rootify),
        simplify_tolerance=100,
        level=0,
        size=112,  # Subtyper trained on 224*0.5 microns
        unit=slide_tools.objects.SizeUnit.MICRON,
        centroid_in_annotation=True,
        img_tfms=resize_wrapper,
        return_index=True,
        backend="openslide",
    )
    print(f"Found {len(ds.samples)} tiles")

    ### Filter ###
    
    to_gray = T.Grayscale()
    normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    def get_white_or_blurry(img, size=60, blurry_thresh=10, white_thresh=210):
        gray = to_gray(img).squeeze(1).numpy()
    
        # grab the dimensions of the image and use the dimensions to
        # derive the center (x, y)-coordinates
        (h, w) = gray.shape[1:]
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
    
        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(gray)
        fftShift = np.fft.fftshift(fft)
    
        # zero-out the center of the FFT shift (i.e., remove low
        # frequencies), apply the inverse shift such that the DC
        # component once again becomes the top-left, and then apply
        # the inverse FFT
        fftShift[:, cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
    
        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude, axis=(1, 2))
        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        blurry = 1 * (mean <= blurry_thresh)
        white_percent = np.sum(gray >= white_thresh, axis=(1, 2)) / (h*w)
    
        white = 1 * (white_percent >= 0.5)
        
        return white_percent, mean, blurry | white
    
    
    ### Model ###

    subtyper = timm.create_model('resnet18')
    subtyper.fc = nn.Sequential(nn.Dropout(0.70), nn.Linear(512, 9))
    state_dict = torch.load("subtyper.pt", map_location="cpu")
    subtyper.load_state_dict(state_dict)
    subtyper.eval()
    
    def get_tile_classifier(x):
        x = x / 255.
        x = normalize(x)
        y = subtyper(x).softmax(1)
        y = y.numpy()
        return y

    all_samples = np.copy(ds.samples)
    unique_slide_ids = np.unique(all_samples[:, 0])

    bs = args.bs
    num_workers = args.num_workers
    save_dir = args.save_dir
    
    if not os.path.exists(save_dir):
        print("Making save_dir:", save_dir)
        os.makedirs(save_dir)
    
    for slide_id in tqdm(unique_slide_ids):
        ds.samples = all_samples[all_samples[:, 0] == slide_id]
        slide = ds.slides[slide_id]
        white_or_blurry = []
        white = []
        blurry = []
        tile_classifier = []
        points = []
        slide_path = slide.image._filename
        label_path = os.path.join(save_dir, os.path.basename(slide_path).split(".")[0] + ".json")
        dl = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=bs, num_workers=num_workers)
        for i, batch in enumerate(tqdm(dl, leave=False)):
            x = batch["img"].permute(0, 3, 1, 2)
            s_ids = batch["slide_idx"]
            assert (s_ids == slide_id).all()
            r_ids = batch["region_idx"]
            
            white_, blurry_, white_or_blurry_ = get_white_or_blurry(x, white_thresh=245)
            
            white.append(white_)
            blurry.append(blurry_)
            white_or_blurry.append(white_or_blurry_)
            tile_classifier.append(get_tile_classifier(x))
            
            x, y, w, h = np.atleast_2d(slide.regions[r_ids]).T
            points.append(np.stack([x + w/2, y + h/2]).T)
        
        points = np.concatenate(points).tolist()
        white = np.concatenate(white).tolist()
        blurry = np.concatenate(blurry).tolist()
        white_or_blurry = np.concatenate(white_or_blurry).tolist()
        tile_classifier = np.concatenate(tile_classifier).tolist()
        label = {
            "white_or_blurry": {"points": points, "values": white_or_blurry},
            "white": {"points": points, "values": white},
            "blurry": {"points": points, "values": blurry},
            "tile_classifier": {"points": points, "values": tile_classifier},
        }
        with open(label_path, "w") as stream:
            json.dump(label, stream)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    with torch.inference_mode():
        main(args)

