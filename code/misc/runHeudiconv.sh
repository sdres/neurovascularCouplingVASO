#!/bin/bash


# step 1
docker run --rm -it \
-v /Users/sebastiandresbach/data/neurovascularCouplingVASO:/base nipy/heudiconv:latest \
-d /base/DICOM/sub-{subject}/ses-{session}/*.IMA \
-o /base/Nifti/ \
-f convertall \
-s 01 \
-ss 01 \
-c none \
--overwrite


# step 2
# make heuristics file


# step 3
docker run --rm -it \
-v /Users/sebastiandresbach/data/neurovascularCouplingVASO:/base nipy/heudiconv:latest \
-d /base/DICOM/sub-{subject}/ses-{session}/*.IMA \
-o /base/Nifti/ \
-f /base/Nifti/code/heudiconvHeuristic.py \
-s 01 \
-ss 01 \
-c dcm2niix \
-b --overwrite
