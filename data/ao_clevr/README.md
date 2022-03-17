## AO-CLEVr Dataset

The dataset we used in our paper: *Compositional recognition with causally-driven embeddings*, Atzmon, Kreuk, Shalit, Chechik, NeurIPS 2020 (Spotlight)
<a href="https://arxiv.org/abs/2006.14610" target="_blank">paper</a> <br>
<a href="" target="_blank">project page</a> <br>


AO-CLEVr is a synthetic-images dataset containing images of "easy" Attribute-Object categories, using the CLEVr framework (Johnson et al. 2017).

AO-CLEVr has attribute-object pairs created from 8 attributes: { red, purple, yellow, blue, green, cyan, gray, brown } and 3 objects {sphere, cube, cylinder}, yielding 24 attribute-object pairs. Each pair consists of 7500 images. Each image has a single object that consist of the attribute-object pair. The object is randomly assigned one of two sizes (small/large), one of two materials (rubber/metallic), a random position and random lightning according to CLEVr defaults. 

### AO-CLEVr files
filename | description
---|---
`features.t7` | Pretrained RESNET-18 features of AO-CLEVr images
`split_metadata_pickles.zip` | A set of pickle files for each split we used. See detailes below
`aoclevr_images.zip` | Raw image files. These were only used to generate the pretrained features
`np_random_state_pickles.zip` | The numpy random number generator (RNG) state for each split. This file is required only when reproducing the results of our model (TBD).  Originally the split were generated on the fly from code. To maintain reproducibility of our reported results we snapshot the random state of the RNG.
`objects_metadata.csv` | Full metadata of the object in each image. The `csv` header is:  `image_filename,split,shape,color,material,size,pixel_coords,3d_coords,image`. Note that `image_filename` and `image` are duplicates

### Loading the metadata of a specific split
The filename of each split has the following format:
`"metadata_ao_clevr__{dataset_variant}_random__comp_seed_{comp_seed}__seen_seed_{seen_seed}__{subset}.pkl"`
its entries are:

1. `dataset_variant` is either `VT` which relates to the (val-test) overlapping split, or `UV` which relates to the non-overlapping split
2. `comp_seed` relates to the compositional part of the split. It consists of a four digit number, formatted as `FFII`, where `FF` indicates the fraction of unseen-test-pairs and `II` indicates the ID of that compositional split (\in {00,01,02})
3. `seen_seed` relates to the seen-samples split. It consists of a single digit number, which indicates the random seed that initialized that split. (\in {0,1,2})
4. `subset` is a string that indicates whether the metadata relates to `train`, `valid` or `test` set.

For example `metadata_ao_clevr__VT_random__comp_seed_4000__seen_seed_1__valid.pkl` contains the information about the validation data of an overlapping split. It is the first (ID=00) compositional split with 40% unseen-test-pairs, and it is the 2nd (seen_seed=1) seen-samples split within that compositional split.

Loading the pickled split-data returns a dictionary with relevant entries.

