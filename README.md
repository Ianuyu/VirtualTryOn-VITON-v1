# 基於可組裝串聯外觀流之虛擬服裝試穿<br>Virtual Try-on Based on Composable Sequential Appearance Flow

## 2024 The Information Technology and Applications Conference (ITAC 2024)
Paper : [Document](docs/paper.pdf)  
Certificate : [Document](docs/論文刊登證明.pdf)  

## 1. Setting Data's Root 
Get a .pkl file contain data's path 
```
python 001_make_Dataset.py
```
If you would like to change the data, weights, output path or other settings,   
you can find them in ```config.py```.

## 2. Training
Start training 
```
python 002_train_gmm.py
```
```
python 003_train_gen.py
```
## 3. Testing 
Start testing
[[Pretrained weights]](https://mega.nz/folder/bMcxjZba#2vixjwZc1KSuA7uMLOdyUg)
```
python 004_test.py
```

## 4. Evaluation
```
=======================  
FID score   : 24.4623  
=======================  
```

## 6. Hardware
The model architectures proposed in this study are implemented using the PyTorchDL framework, and training is conducted on hardware featuring an Intel® Core™ i7-12700 CPU and Nvidia RTX 3060 12GB graphics processing unit (GPU).
