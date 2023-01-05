
"""

Polish manually edited MRI white & gray matter segmentations.

"""

import os
import nibabel as nb
import numpy as np
from scipy.ndimage import morphology, generate_binary_structure
from scipy.ndimage import gaussian_filter

# Set data path
DATADIR = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

# Set subjects to work on
subs = ['sub-05']


for sub in subs:

    subFolder = f'{DATADIR}/{sub}/ses-01/anat/upsample'
    # subFolder = f'{DATADIR}/{sub}/segmentation'
    # Segmentation file
    FILE = f'{subFolder}/gmwm-LH2.nii.gz'

    # Integer labels for tissue classes
    WM = 2
    GM = 3

    # Output suffix
    SUFFIX = "polished"

    # =============================================================================
    # Load data
    nii = nb.load(FILE)
    data = np.asarray(nii.dataobj)
    data = np.pad(data, 1, mode="reflect")  # to prevent data edge artifacts

    # -----------------------------------------------------------------------------
    # Separate white matter
    wm = data == WM

    # [Step-01] Dilate (to preserve thin bridges frequent in white matter)
    struct = generate_binary_structure(3, 1)  # 1 jump neighbourbhood
    wm = morphology.binary_dilation(wm, structure=struct, iterations=1)

    # [Step-02] Smooth
    FWHM = 2  # Full width half maximum of Gaussian kernel. In voxel size units.
    SIGMA = FWHM / 2.35482004503  # Convert to filter standard deviation
    wm = gaussian_filter(wm.astype(float), sigma=SIGMA, mode="reflect")
    wm = wm > 0.5

    # [Step-03] Erode (to go back to original white matter average thickness)
    wm = morphology.binary_erosion(wm, structure=struct, iterations=1)

    # -----------------------------------------------------------------------------
    # Separate white matter toether with gray matter
    wmgm = data == GM
    wmgm = wmgm + wm

    # [Step-01] Erode (to keep kissing gyri as separate as possible)
    wmgm = morphology.binary_erosion(wmgm, structure=struct, iterations=1)

    # [Step-02] Smooth & re-binarize
    wmgm = gaussian_filter(wmgm.astype(float), sigma=SIGMA, mode="reflect")
    wmgm = wmgm > 0.5

    # [Step-03] Dilate (to go back to original outer gray matter average thickness)
    wmgm = morphology.binary_dilation(wmgm, structure=struct, iterations=1)

    # -----------------------------------------------------------------------------
    # Reform the segmentation file
    final = (data != 0).astype(int)
    final[wmgm != 0] = 3
    final[wm != 0] = 2

    # Trim padded elements
    final = final[1:-1, 1:-1, 1:-1]

    # Save as nifti
    SUFFIX = "polished"
    basename, ext = nii.get_filename().split(os.extsep, 1)
    out = nb.Nifti1Image(final.astype(int), header=nii.header, affine=nii.affine)
    nb.save(out, "{}_{}.{}".format(basename, SUFFIX, ext))

    print("Finished. Don't forget the check the result carefully!.")
