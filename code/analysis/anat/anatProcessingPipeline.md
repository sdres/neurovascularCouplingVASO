# Anatomical processing pipelin

Here, I am collecting notes for the processing of the anatomical data (MP2RAGE and MEGRE).

1. averaging MP2RAGE images
- (bias field correction)
- brain extraction guided by inv-2
- (editing using segmentator)
- (cropping the data)
- Upsampling
- coregistration of inv-2 images
  - Load ants registration matrices in itksnap and tinker around with parameters
  - Save transformation matrices in repository
- applying transform to uni images
- averaging

- consider downsampling, running segmentation and then upsample segmentation again.

2. Segmentation
- Define sphere in native space and repeat cropping and upsampling
- Running automatic segmentation on average uni image
- manual segmentation edits
- sulci enhancing filter

3. MEGRE processing


## Open questions

- At what resolution should we do the segmentation?




segmentation times
sub-05 ~6h
sub-06 45 minutes 45 minutes

segmentation centers
sub-05
sub-06 LH itksnap whole brain 88 45 151


crop sphere
sub-06
fslroi sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X_sphere.nii.gz sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_N4cor_brain_crop_ups4X_sphere_crop.nii.gz 271 162 7 162 31 159

fslroi sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_brain_pveseg_corrected_crop_ups4X_sphere.nii.gz sub-06_ses-01_uni_part-mag_run-01_MP2RAGE_brain_pveseg_corrected_crop_ups4X_sphere_crop.nii.gz 271 162 7 162 31 159
