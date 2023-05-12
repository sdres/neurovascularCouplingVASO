"""Read BrainVoyager vmr and export nifti."""

import os
import nibabel as nb
import bvbabel
import pprint

SUBS = ['sub-05']
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

for sub in SUBS:

    folder = f'{ROOT}/rawdata_bv/{sub}/ses-01/anat'

    FILE = f'{folder}/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_denoised_IIHC_pt5_ACPC.vmr'

    # =============================================================================
    # Load vmr
    header, data = bvbabel.vmr.read_vmr(FILE)

    # See header information
    pprint.pprint(header)

    # Load affine of original
    original = f'{ROOT}/{sub}/ses-01/anat/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor.nii.gz'
    affine = nb.load(original).affine

    # Export nifti
    basename = FILE.split(os.extsep, 1)[0]
    outname = "{}_bvbabel.nii.gz".format(basename)
    # img = nb.Nifti1Image(data, affine=np.eye(4))
    img = nb.Nifti1Image(data, affine=affine)
    nb.save(img, outname)

    print("Finished converting.")

    # moving = f'{FOLDER}/sub-05_ses-01_uni_part-mag_run-01_MP2RAGE_denoised_IIHC_pt5_ACPC_test.nii.gz'
    # reference = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives/sub-05/ses-01/anat/upsample/sub-05_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
    # matrix = f'{FOLDER}/sub-05_ses-01_uni_part-mag_run-01_MP2RAGE_denoised_IIHC_pt5_ACPC_bvbabel_registered-sub-05_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop.txt'
    #
    # command = f'c3d {reference} {moving} -reslice-itk {matrix} -o test.nii'
    # print(command)