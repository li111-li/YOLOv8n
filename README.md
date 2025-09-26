# HSA Test Strip Colorimetric Recognition Using YOLOv8n

## Introduction
This repository provides the dataset, code, and trained YOLOv8n model weights for rapid recognition of HSA concentrations from colorimetric test strips.

## Enviroment
The code is developed using Python 3.10 on Windows 11. NVIDIA GPUs are needed. The code is developed and tested using an NVIDIA GeForce RTX 3090 GPU.
- PyTorch 2.0.1 + cu113
- TorchVision 0.15.2
- TorchAudio 2.0.2
- Numpy 2.1.1

## Prepare datasets
The dataset includes images of HSA test strips at varying concentrations.<br> 
The dataset is organized as follows:<br> 
```
datasets/
├── images/
│ ├── train/
│ │ ├── img_0-1.jpg
│ │ ├── img_0-2.jpg
│ │ └── ...
│ ├── val/
│ │ ├── img_0-1.jpg
│ │ ├── img_0-2.jpg
│ │ └── ...
│ └── test/
│   ├── img_0-1.jpg
│   ├── img_0-2.jpg
│   └── ...  
├── labels/
│ ├── train/
│ │ ├── img_0-1.jpg
│ │ ├── img_0-2.jpg
│ │ └── ...
│ ├── val/
│   ├── img_0-1.jpg
│   ├── img_0-2.jpg
│   └── ...
├──data.yaml
```
Place your training 、validation and test images in `datasets/images/train、 datasets/images/val and datasets/images/test`.<br>
Place corresponding label files in YOLO format (class x_center y_center width height) in `datasets/labels/train and datasets/labels/val`.<br>
This `data.yaml` file contains the dataset paths and class information. Adjust paths in data.yaml if necessary.

## Training
To start training the YOLOv8n model, open a terminal and run the following command:<br> 
```
yolo train data=datasets\data.yaml model=yolov8n.pt epochs=150 lr0=0.001 batch=4
```
`data=datasets\data.yaml`：The training command specifies the dataset yaml file <br> 

`model=yolov8n.pt`: the pre-trained YOLOv8n weights 

`epochs=150 lr0=0.001 batch=4`: the number of training epochs (epochs, set to 150), the initial learning rate (lr0, set to 0.001), and the batch size (batch, set to 4).<br> 

After training, the evaluation results, including predicted images and summary metrics, will be saved in the `runs/train/` folder.

### Testing
After training, you can evaluate the model on the test dataset using the following command:<br> 
```
yolo predict model=runs/detect/train/weights/best.pt source=datasets/images/test
```
`model=runs/detect/train/weights/best.pt`: uses the trained model weights (best.pt) obtained from training.<br>

`source=datasets/images/test`: specifies the path to the test images on which predictions will be made.<br>

This command will output predicted results with bounding boxes and save them in the `runs/predict/`folder.<br>
