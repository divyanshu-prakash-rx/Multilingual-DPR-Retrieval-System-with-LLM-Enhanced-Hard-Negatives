#!/bin/bash

python data_processing/download/msmarco_dev.py
python data_processing/download/download_and_prepare_tydi_beir_format.py
python data_processing/download/download_mmarco.py