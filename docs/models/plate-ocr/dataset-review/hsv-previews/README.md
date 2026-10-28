# HSV lighting previews

Derived from the three source crops listed in `manifest.csv`; generated only for visual review, not training. H is held constant so the sheet isolates saturation/value changes. ImageMagick HSV stores H/S/V in its RGB channel slots after conversion: G is S and B is V.

Profiles: brighter S×0.95/V×1.15; dim S×0.85/V×0.75; dark S×0.80/V×0.55; overexposed S×0.75/V×1.30. Values above the range clip. These are global HSV transforms; they do not simulate localized headlight glare, spatial shadows, sensor noise, or motion blur. Candidate settings are illustrative, not validated hyperparameters.
