# Data placement

- Put raw `uint16` time-series files in `raw/` before preprocessing.
- Short-acquisition quantitative MAT files are written to or placed in
  `quant_maps/short_acquisition/`.
- Matched long-acquisition reference MAT files are written to
  `quant_maps/long_reference/` and are required for training, but not inference.

Each quantitative MAT file must contain a struct named `maps`. See the main
README for the required fields and acquisition assumptions.

One paired OA-cell example is included so the full package can be tested
immediately. Additional files can be added to the same folders and will be
processed in batch.
