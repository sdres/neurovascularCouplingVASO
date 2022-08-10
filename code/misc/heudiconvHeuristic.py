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

    info = {vasoMagn:[],vasoPhs:[],boldMagn:[],boldPhs:[]}

    for idx, s in enumerate(seqinfo):
        if s.series_files > 280:
            if ('S00_M' in s.series_description):
                info[vasoMagn].append(s.series_id)
            if ('S00_P' in s.series_description):
                info[vasoPhs].append(s.series_id)
            if ('S01_M' in s.series_description):
                info[boldMagn].append(s.series_id)
            if ('S01_P' in s.series_description):
                info[boldPhs].append(s.series_id)
    return info
