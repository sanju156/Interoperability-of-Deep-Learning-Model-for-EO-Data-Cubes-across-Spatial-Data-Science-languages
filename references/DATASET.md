# Dataset References

This project uses the BreizhCrops time series dataset for crop-type mapping, prepared by:

Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., and Körner, M.: BREIZHCROPS: A TIME SERIES DATASET FOR CROP TYPE MAPPING, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci., XLIII-B2-2020, 1545–1551, https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020, 2020.

[GitHub Repository](https://github.com/dl4sits/BreizhCrops)

## Data Access
Original dataset and preprocessing code: https://github.com/dl4sits/BreizhCrops

* Geographic region: Brittany, France (NUTS3 level) - covering 27,200 km²

* Data type: HDF5 files containing Sentinel-2 band values with filed ids and crop lables.

* Temporal coverage: 2017 growing season (1 year) – 5 days temporal revisit.

* Each time series was randomly sampled to a fixed length of 45 observations.

* Number of crop classes : 9

* Input shape for the model: (N × 13 × 45) -> (Batch,Bamds, Sequence length)

* Class imbalance visible (barley/wheat dominant vs sunflower/nuts)


## License & Attribution

* The original dataset is publicly available via python package and subject to their data usage policies.

* If you use this dataset in your work, please credit:
BreizhCrops for the crop-type data
Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., and Körner, M. for the preprocessed dataset and baseline model code.