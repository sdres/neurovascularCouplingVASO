#!/bin/bash


# step 1
docker run --rm -it \
-v /Users/sebastiandresbach/data/neurovascularCouplingVASO:/base nipy/heudiconv:latest \
-d /base/DICOM/sub-{subject}/ses-{session}/*/*.dcm \
-o /base/Nifti/ \
-f convertall \
-s 05 \
-ss 02 \
-c none \
--overwrite


# step 2
# make heuristics file


# step 3
docker run --rm -it \
-v /Users/sebastiandresbach/data/neurovascularCouplingVASO:/base nipy/heudiconv:latest \
-d /base/DICOM/sub-{subject}/ses-{session}/*/*.dcm \
-o /base/Nifti/ \
-f /base/Nifti/code/heudiconvHeuristic.py \
-s 08 \
-ss 03 \
-c dcm2niix \
-b --overwrite
