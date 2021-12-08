# Data generation and datasets

There are several datasets and data generations scripts in this project:

* `sentinel-2_artificial` is a Sentinel 2 dataset created by KP Labs. The high
  resolution images are the original images from the Sentinel mission. The low
  resolution counterparts are made from the existing using bicubic
  interpolation and some other transformations (e.g. exposure, noise, blur).
  However, these LRs are not directly used in this project. They serve as
  placeholders. This dataset is used to export the 'degraded' Sentinel datasets.
* `proba-v` is the original Proba V dataset from the Proba V challange. It
  includes a 'test' subdirectory without HR images. It is used to create
  modified Proba datasets.
* `proba-v_registered` includes Proba-V image pairs registered using phase
  correlation. To registered HR and LR images in pairs the HR image was
  *multiplied* (copied). Thus, the directory structure inside scenes is a bit
  different then in the original set. Moreover, a new test set was created from
  a subset of train images. The test subset in registered Proba hold both HR
  and LR images.
* `proba_v_registered_a` and `proba_v_registered_b` are subsets of the
  registered Proba V dataset. They include the same test subset as the
  `proba_v_registered`; however, the train subsets are split in half. The 'a'
  set is selected to train augmentation networks, the 'b' set is chosen for
  degradation and SR training.
