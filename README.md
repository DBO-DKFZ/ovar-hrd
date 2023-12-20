Code for HRD prediction on ovarian cancer
-----------------------------------------

This repo is a one-time code drop of a publication under review.

Installation
------------

````
git clone https://github.com/DBO-DKFZ/ovar-hrd.git
cd ovar-hrd
mamba env create -f environment.yml -n ovar_hrd
mamba activate ovar_hrd
````


Pipeline
--------
We will go through our pipeline with two WSIs from TCGA you have to download and place into [sample/slides](sample/slides):  
  - `TCGA-13-A5FT-01Z-00-DX1.2B292DC8-7336-4CD9-AB1A-F6F482E6151A.svs`
  - `TCGA-13-A5FU-01Z-00-DX1.9AD9E4B9-3F87-4879-BC0F-148B12C09036.svs`

We omit the very first step, which is tissue detection with QuPath. The corresponding script is [0_detect/tissue-detection.groovy](0_detect/tissue-detection.groovy). The output of this script is already included in [sample/annotations](sample/annotations).

### Filter Tiles
This step will go through the WSIs and compute *blurryness*, *backgroundness* and *tissue type* for each tile within the detected tissue. Output is saved in [sample/tile-filter](sample/tile-filter).
````
cd 1_filter
python filter.py --csv ../sample/tcga_sample.csv \
    --root .. \
    --save_dir ../sample/tile-filter \
    --bs 32 \
    --num_workers 8
````

### Extract Tile Features
This step will extract features from each tile with some pretrained encoder and use the filter we computed above. Output is saved to [sample/tile-features/resnet18-camelyon_catavgmax](sample/tile-features/resnet18-camelyon_catavgmax).

````
cd 2_extract
python extract.py --encoder_name resnet18-camelyon_catavgmax \
    --with-filter white_blurry_tumor \
    --prediction_dir ../sample/tile-features \
    --csv_predict "sample/tcga_sample.csv" \
    --root .. \
    --regions_centroid_in_annotation true \
    --regions_size 112 \
    --regions_unit micron \
    --slide_interpolation linear \
    --slide_simplify_tolerance 100 \
    --tfms_test "resize,normalize,gpu" \
    --column_label label \
    --column_slide slide \
    --column_annotation annotation \
    --image_size 224 \
    --regions_return_index true \
    --accelerator cpu \
    --batch_size 64 \
    --num_workers 8 \
    --name resnet18-camelyon_catavgmax
````

### Predict Slide Score
This step will use the extracted tile features and predict an HRD score per WSI.
````
cd 3_predict
python predict.py --preds_folder ../sample/tile-features/resnet18-camelyon_catavgmax/tcga_sample \
    --model_path "ago_train-mannheim_train_seed=26694295_auroc=0.78250_epoch=74.pt" \
    --save_path ../sample/hrd_preds.csv
````
