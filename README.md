# Super-Resolution-Flow-Field-Reconstruction-for-Extreme-Reactive-Flow-System

This code was used to implement: Super-Resolution-Flow-Field-Reconstruction-for-Extreme-Reactive-Flow-System (2023)

## Dependencies
```
pip install -r requirements.txt
```

## Dataset

The dataset is calculated by TurfSIM of SCP around ~25GB.

### Downloading data
1. The source files are listed in the **source** directory. Open the following link to download .dat source files:

2. After the tif files are downloaded, run **generate_data.py** (under **source** directory). This will process the source files and construct an index for the dataset.

## Example

For distributed training with 1 GPU, batch size of 64:
```
python train.py --batchSize 64 --nEpochs 200
```
For a complete test set evaluation:
```
python test.py --dic #path_to_model
```
