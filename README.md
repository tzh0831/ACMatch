# Dataset Info

Public Dataset refs:

- SUN RGB-D: https://rgbd.cs.princeton.edu/
- NYUv2: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html

Prepare datasets to have following structure:

```
  NYUV2
    100/
      test.txt
      train_label.txt
    199/
      test.txt
      train_label.txt
    398/
      test.txt
      train_label.txt
      train_unlabel.txt
    depths/
      ...
    images/
      ...
    labels/
      ...
    labels40/
      ...
    preprocess.py
    README.md
    splits.mat
    train.txt
    val.txt

  SUN RGB-D
    330/
      test.txt
      train_label.txt
      train_unlabel.txt
    661/
      test.txt
      train_label.txt
      train_unlabel.txt
    1322/
      test.txt
      train_label.txt
    kv1/
      b3dodata/
        img_**/
          annotation
          annotation2D3D
          annotation2Dfinal
          annotation3D
          annotation3Dfianl
          annotation3Dlayout
          depth
          depth_bfx
          extrinsics
          fullres
          image
          intrinsics.txt
          label
          scene.txt
          seg.mat
        ...
      NYUdata
        NYU**/
          目录文件同上
        ...
    kv2/
      align_kv2/
        2014-12-**/
          目录文件同上
        ...
      kinect2data/
        00**/
          目录文件同上，多了一个status文件夹
    realsense
    test_depth.txt
    test_label.txt
    test_rgb.txt
    train_depth.txt
    train_label.txt
    train_rgb.txt
    xtion
```

Training:

```python
  python train/ACT-Unimatch_Fp.py
```
```
  python train/ACT-Unimatch.py
```