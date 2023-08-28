#!/usr/bin/python
import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):

    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    vasoMagn = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-stimulation_run-0{item:01d}_part-mag_cbv')
    vasoPhs = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-stimulation_run-0{item:01d}_part-phase_cbv')
    boldMagn = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-stimulation_run-0{item:01d}_part-mag_bold')
    boldPhs = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-stimulation_run-0{item:01d}_part-phase_bold')

    # MP2RAGE
    mp2rageInv1Mag = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_inv-1_part-mag_run-0{item:01d}_MP2RAGE')
    mp2rageInv1Phase = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_inv-1_part-phase_run-0{item:01d}_MP2RAGE')
    mp2rageInv2Mag = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_inv-2_part-mag_run-0{item:01d}_MP2RAGE')
    mp2rageInv2Phase = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_inv-2_part-phase_run-0{item:01d}_MP2RAGE')
    mp2rageUniMag = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_uni_part-mag_run-0{item:01d}_MP2RAGE')
    mp2rageUniPhase = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_uni_part-phase_run-0{item:01d}_MP2RAGE')
    mp2rageT1Mag = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_t1_part-mag_run-0{item:01d}_MP2RAGE')
    mp2rageT1Phase = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_t1_part-phase_run-0{item:01d}_MP2RAGE')

    info = {vasoMagn: [],
            vasoPhs: [],
            boldMagn: [],
            boldPhs: [],

            # MP2RAGE
            mp2rageInv1Mag: [],
            mp2rageInv1Phase: [],
            mp2rageInv2Mag: [],
            mp2rageInv2Phase: [],
            mp2rageUniMag: [],
            mp2rageUniPhase: [],
            mp2rageT1Mag: [],
            mp2rageT1Phase: []
            }

    for idx, s in enumerate(seqinfo):
        if s.series_files > 210:
            if ('S00_M' in s.series_description):
                info[vasoMagn].append(s.series_id)
            if ('S00_P' in s.series_description):
                info[vasoPhs].append(s.series_id)
            if ('S01_M' in s.series_description):
                info[boldMagn].append(s.series_id)
            if ('S01_P' in s.series_description):
                info[boldPhs].append(s.series_id)
        # MP2RAGE
        if ('INV1' in s.series_description):
            info[mp2rageInv1Mag].append(s.series_id)
        if ('INV2' in s.series_description):
            info[mp2rageInv2Mag].append(s.series_id)
        if ('UNI' in s.series_description):
            info[mp2rageUniMag].append(s.series_id)
        if ('T1' in s.series_description):
            info[mp2rageT1Mag].append(s.series_id)

    return info
