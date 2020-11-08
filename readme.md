# Data Preparation:

Download the dataset. 

For comparison experiments with VPE, the dataset can be downloaded from: [VPE repo](https://github.com/mibastro/VPE). 

# Dependence
Our Python version is 3.7.4. ```requirements.txt``` has been provided in the code folder, with the following content:
```
matplotlib==3.1.1
scipy==1.1.0
torchvision==0.5.0
numpy==1.17.2
torch==1.4.0
Pillow==7.0.0
```
You can do ``` pip install -r requirements.txt ``` for installing all the dependencies.

Please switch to scipy 1.1.0 by
```
pip install scipy==1.1.0
```
If you see errors like this
```
AttributeError: module 'scipy.misc' has no attribute 'imread'
```

# Extract the dataset
```
tar -xzvf db.tar.gz 
```
The code folder should look like
```
    
       code
        |
        |__ *.py
        |__ config.json
        |__ log_ours
        |__ ...
        |__ db
            |__ belga
            |__ flickr32
            |__ toplogo10
            |__ GTSRB
            |__ TT100K
            |__ exp_list
            |__ ASL
```

# Reproduce the result
We can start four experiments by:

## Belga->Flicker32 
```
python ours_flickr32.py

```
Data log folder: ```results_belga2flickr```

## Belga->Toplogo
```
python ours_toplogo.py

```
Data log folder: ```results_belga2toplogo```

## GTSRB->GTSRB
```
python ours_gtsrb.py

```
Data log folder: ```results_gtsrb```

## GTSRB->TTK100
```
python ours_TT100K.py

```
Data log folder: ```results_gtsrb2TT100K```

# References

```
Kim, Junsik, et al. "Variational prototyping-encoder: One-shot learning with prototypical images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
```
As our code is based on VPE, our code is also improved from their paper code. [VPE repo](https://github.com/mibastro/VPE)

