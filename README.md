# KAIST_CS492_DeepImplicitTemplates
KAIST CS492(A) Assignment

This repository is an implementation for [Deep Implicit Templates](http://www.liuyebin.com/dit/dit.html). 
And the paper is [Deep Implicit Templates](https://arxiv.org/abs/2011.14565). 

## Requirements
* Ubuntu(v18.04)
* numpy // 1.22.3
* Pytorch // 1.9.0+cu111
* sklearn // 0.24.1
* ninja // 1.10.2.3
* pathos // 0.2.8
* trimesh // 3.10.2
* skimage // 0.19.2
* pyrender // 0.1.45
* scipy // 1.8.0
* mesh_to_sdf
* plyfile
* easydict
* tqdm
* os

## Data Preprocessing

Our preprocessing code is highly based on mesh2sdf project and [ShapeGAN](https://github.com/marian42/shapegan).


```
python prepare_data_dir-pn.py
```

## Implementataion
 A benchmark dataset of our project is [ShapeNetCore.v2](https://shapenet.org/). And you can download preprocessed datasets(SDF files) and pre-trained model from this [Gdrive Link](https://drive.google.com/drive/folders/1lshhJJNP_lbVG9BQjM0eME7x3JvPyEME?usp=sharing).
 
```
git clone https://github.com/Goya-Studio/KAIST_CS492_DeepImplicitTemplates.git
cd KAIST_CS492_DeepImplicitTemplates
```

1. Deep Implicit Templates.ipynb  : Implement and train DIT model.
2. Reconstruction.ipynb           : Reconstruct a shape using trained model.(latent code train again.)
3. Evaluation.ipynb               : Evaluate the result by EMD, CD metrics.
- You need to download [ShapeNetCore.v2](https://shapenet.org/) for evaluation.
- If you download the ShapeNet data, please put the data in gt folder.

   ![image](https://user-images.githubusercontent.com/74032553/172082907-fd6e7100-b28e-4a6a-aadf-33cbfb2b1749.png)


## Acknowledgements

This repository is based on [DeepSDF](https://github.com/facebookresearch/DeepSDF) and [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates). And we use the [mesh2sdf](https://github.com/marian42/mesh_to_sdf) project to preprocess the data. Thank you all!

## Reference Project
* [DeepSDF](https://github.com/facebookresearch/DeepSDF)
* [Deep Implicit Templates](https://github.com/ZhengZerong/DeepImplicitTemplates)
* [mesh2sdf](https://github.com/marian42/mesh_to_sdf)
* [ShapeGAN](https://github.com/marian42/shapegan)
* [ShapeNetCore.v2](https://shapenet.org/)

## [NOTE]
* If you have a problem about 'pip install easydict', please git clone this [project](https://github.com/makinacorpus/easydict).


## Contact
- YONGMIN: kymin1002@kaist.ac.kr
- Olivia: oo2703@kaist.ac.kr
