"""Read BrainVoyager vmr and export nifti."""

import os
import nibabel as nb
import bvbabel
import pprint
import subprocess

subs = ['sub-08']
ROOT = '/Users/sebastiandresbach/data/neurovascularCouplingVASO/Nifti/derivatives'

for sub in subs:

    folder = f'{ROOT}/rawdata_bv/{sub}/ses-01/anat'

    inFile = f'{folder}/v1-LH.vmr'

    # =============================================================================
    # Load vmr
    header, data = bvbabel.vmr.read_vmr(inFile)

    # See header information
    pprint.pprint(header)

    # Load affine of original
    original = f'{ROOT}/{sub}/ses-01/anat/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor.nii.gz'
    affine = nb.load(original).affine

    # Export nifti
    basename = inFile.split(os.extsep, 1)[0]
    outname = "{}_bvbabel.nii.gz".format(basename)
    # img = nb.Nifti1Image(data, affine=np.eye(4))
    img = nb.Nifti1Image(data, affine=affine)
    nb.save(img, outname)

    print("Finished converting.")

    print("Registering to high resolution anatomy")

    moving = outname
    reference = f'{ROOT}/{sub}/ses-01/anat/upsample/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'
    matrix = f'{folder}/{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_denoised_IIHC_pt5_ACPC_bvbabel_registered-{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop.txt'
    outFile = f'{folder}/v1_registered-{sub}_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X.nii.gz'

    command = f'c3d {reference} {moving} -reslice-itk {matrix} -o {outFile}'
    subprocess.run(command, shell=True)

    print('Binarizing')
    command = f'fslmaths {outFile} -bin {outFile}'
    subprocess.run(command, shell=True)
