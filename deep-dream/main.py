import torch
from PIL import Image
import os
from dreamify import dreamify
import cv2
import numpy as np

OCT_SCALE = 1.3
NUM_OCTAVES = 10

for im_path in os.listdir('in'):
	with Image.open(f'in/{im_path}') as img:
		w, h = img.size
		w = w / 1.3 ** NUM_OCTAVES
		h = h / 1.3 ** NUM_OCTAVES
		img = img.resize((int(w), int(h)))
		img = np.array(img)
		
		# Dreamify
		for octave in range(NUM_OCTAVES):
			w *= OCT_SCALE
			h *= OCT_SCALE
			img = cv2.resize(img, (int(w), int(h)))
			
			n_iters = 5
			if octave == NUM_OCTAVES - 1:
				n_iters = 20
			img = dreamify(img, n_iters, 0.05)

		img = Image.fromarray(img)

		img.save(f'out/{im_path}')