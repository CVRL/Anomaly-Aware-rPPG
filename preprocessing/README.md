## Preprocessing a Dataset
1.) To crop all of the videos to face landmarks from MediaPipe and downscale to 64x64 pixels for the RPNet 3DCNN model:

```
python make_dataset.py <path-to-folder-of-videos> <path-to-folder-of-npzs>
```

2.) You can now extract the CHROM and POS baselines from the videos

```
python get_baselines.py <path-to-folder-of-videos> <path-to-folder-of-npzs> <path-to-baseline-outputs> --waves <optional-path-to-folder-of-ground-truth>
```

3.) Then make the metadata file which contains paths to all of the preprocessed data:

```
python make_metadata.py <path-to-folder-of-npzs> <path-to-metadata-csv>
```
