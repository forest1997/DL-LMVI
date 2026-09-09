"""
Improved cell-outline variant of the migrated UNet++ ResNet34 pipeline.

Only the Cellpose whole-cell configuration is changed:
  - legacy Cellpose ``cyto`` model used by the cell-outline reference script
  - green fluorescence channel only
  - 80 px cell diameter, selected after visual comparison on OA, 2dg, HeLa,
    and HepG2 representative images

The UNet++ nucleus/nucleolus model, lipid logic, masks, and visualizations are
provided by the original migrated pipeline without modification.
"""

import importlib.util
import os


PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_SCRIPT = os.path.join(PIPELINE_ROOT, "UNet++-ResNet34_test_and_mask.py")

spec = importlib.util.spec_from_file_location("unetpp_resnet34_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load base pipeline: {BASE_SCRIPT}")

pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

# Cellpose's channel numbering is 1-based for RGB: 2 = green, 0 = none.
pipeline.CELLPOSE_MODEL_TYPE = "cyto"
pipeline.CELLPOSE_CHANNELS = [2, 0]
pipeline.CELLPOSE_DIAMETER = 80


if __name__ == "__main__":
    print("Cell-outline improvement: model=cyto, channel=green, diameter=80 px")
    pipeline.main()
