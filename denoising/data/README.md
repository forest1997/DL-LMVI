# Included quantitative-map example

`quant_maps/short_acquisition` contains one 48-frame OA-cell map.
`quant_maps/long_reference` contains its matching 512-frame reference.
Both use `powerMode="amplitudeSquared"`: V is already divided by N squared.
Do not use v1 legacy V values as v2 inputs. H is in Hz; V is linear.

The example is in the fixed 35-image evaluation list. It is enough for inference
but not training. The full training dataset is not included.

`raw` is intentionally empty. Optional raw preprocessing assumes headerless
little-endian uint16, 800 x 550 pixels per frame, and 100 Hz. Check these settings
before preprocessing new acquisitions. The 48-frame window is the beginning
of the same file used for the 512-frame reference.
