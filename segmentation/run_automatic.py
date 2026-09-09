"""Packaging-only launcher: invokes the unmodified improved inference script."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='Run the verified cyto + U-Net++ pipeline on one image folder.')
    parser.add_argument('--input',help='Folder containing RGB images (non-recursive)')
    parser.add_argument('--output',help='New output folder; must be different from input')
    args=parser.parse_args()
    source=Path(args.input or input('Input image folder: ').strip().strip('"')).resolve()
    if not source.is_dir(): raise SystemExit('Input folder not found.')
    destination=Path(args.output or input('Output folder (empty = package/Results): ').strip().strip('"') or ROOT/'Results').resolve()
    if destination==source: raise SystemExit('Output must be different from input.')
    pipe=ROOT/'Automatic_Segmentation'
    for weight in [pipe/'trained_models/best_unetpp_resnet34.pth',pipe/'cellpose_models/cytotorch_0']:
        if not weight.is_file(): raise SystemExit(f'Missing model weight: {weight}')
    env=os.environ.copy()
    env.update(CELLSEG_PREDICT_DATA_ROOT=str(source.parent),CELLSEG_ORIGINAL_FOLDER=source.name,
               CELLSEG_OUTPUT_ROOT=str(destination),CELLPOSE_LOCAL_MODELS_PATH=str(pipe/'cellpose_models'),
               PYTHONUTF8='1',PYTHONUNBUFFERED='1')
    return subprocess.call([sys.executable,str(pipe/'UNet++-ResNet34_cellpose_outline_improved.py')],env=env,cwd=str(ROOT))

if __name__=='__main__': sys.exit(main())
