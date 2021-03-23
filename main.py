#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:53:17 2020

@author: andrew
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from trainer import train_l1, train_cgan
from util import ingest_kazr, get_kazr_sample_inds, ingest_csapr, make_kazr_test_sets, make_csapr_test_set, create_sample_sets
cases = ['outage','downfill','blockage']

#preprocess the datasets:
ingest_kazr()
get_kazr_sample_inds()
ingest_csapr()
make_kazr_test_sets()
make_csapr_test_set()
for case in cases:
    create_sample_sets(case)

#train the CNNs (this loop trains 6 CNNs so break this up as needed)
for case in cases:
    train_l1(case)
    train_cgan(case)

#evaluate the results:
from evaluation import compute_error_metrics
for case in cases:
    compute_error_metrics(case)