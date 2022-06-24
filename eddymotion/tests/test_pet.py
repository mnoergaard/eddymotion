#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:28:00 2022

@author: martinnorgaard
"""

from eddymotion.dmri import load
import numpy as np
from eddymotion.viz import plot_dwi
from eddymotion.model import AveragePETModel
from nipype.interfaces.ants import Registration
from pathlib import Path
from tempfile import mkdtemp

grad = np.ones((21, 4))
gradient_file = '/Users/martinnorgaard/Dropbox/Mac/Documents/GitHub/eddymotion/eddymotion/tests/data/pet_gradient_file.txt'
np.savetxt(gradient_file, grad)

PET = load('/Users/martinnorgaard/Dropbox/Mac/Documents/GitHub/eddymotion/eddymotion/tests/data/sub-01_ses-baseline_pet.nii.gz',gradient_file)

data_train, data_test = PET.logo_split(10)

model = AveragePETModel(PET.gradients, th_low=0, th_high=3, bias=False)

model.fit(data_train[0])
predicted = model.predict(data_test[1])

plot_dwi(predicted, PET.affine, gradient=data_test[1]);

tempdir = Path(mkdtemp())

# The fixed image is our prediction
fixed_path = tempdir / "fixed.nii.gz"
PET._to_nifti(predicted, dmri_dataset.affine, fixed_path)

# The moving image is the left-out DW map
moving_path = tempdir / "moving.nii.gz"
PET._to_nifti(data_test[0], dmri_dataset.affine, moving_path)



