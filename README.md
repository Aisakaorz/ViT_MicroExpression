# ViT_MicroExpression

Use ViT series on micro-expression recognition

## FacePretreatment_dlib

### Dlib Models

| Model                                                        |                                  Download                                   |
|:-------------------------------------------------------------|:---------------------------------------------------------------------------:|
| dlib_face_recognition_resnet_model_v1.dat (please unzip bz2) | [here](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) |
| shape_predictor_68_face_landmarks.dat (please unzip bz2)     |   [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)   |

## T2T_ViT

### T2T-ViT Models

| Model                      | T2T Transformer | Top1 Acc | params | MACs  |                                              Download                                              |
|:---------------------------|:---------------:|:--------:|:------:|:-----:|:--------------------------------------------------------------------------------------------------:|
| T2T-ViT-14                 |    Performer    |   81.5   | 21.5M  | 4.8G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.5_T2T_ViT_14.pth.tar)  |
| T2T-ViT-19                 |    Performer    |   81.9   | 39.2M  | 8.5G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.9_T2T_ViT_19.pth.tar)  |
| T2T-ViT-24                 |    Performer    |   82.3   | 64.1M  | 13.8G | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.3_T2T_ViT_24.pth.tar)  |
| T2T-ViT-14, 384            |    Performer    |   83.3   | 21.7M  |       | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/83.3_T2T_ViT_14.pth.tar)  |
| T2T-ViT-24, Token Labeling |    Performer    |   84.2   |  65M   |       | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/84.2_T2T_ViT_24.pth.tar)  |
| T2T-ViT_t-14               |   Transformer   |   81.7   | 21.5M  | 6.1G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/81.7_T2T_ViTt_14.pth.tar) |
| T2T-ViT_t-19               |   Transformer   |   82.4   | 39.2M  | 9.8G  | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.4_T2T_ViTt_19.pth.tar) |
| T2T-ViT_t-24               |   Transformer   |   82.6   | 64.1M  | 15.0G | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/82.6_T2T_ViTt_24.pth.tar) |

The 'T2T-ViT-14, 384' means we train T2T-ViT-14 with image size of 384 x 384.

The 'T2T-ViT-24, Token Labeling' means we train T2T-ViT-24
with [Token Labeling](https://github.com/zihangJiang/TokenLabeling).

The three lite variants of T2T-ViT (Comparing with MobileNets):

| Model      | T2T Transformer | Top1 Acc | params | MACs |                                             Download                                              |
|:-----------|:---------------:|:--------:|:------:|:----:|:-------------------------------------------------------------------------------------------------:|
| T2T-ViT-7  |    Performer    |   71.7   |  4.3M  | 1.1G | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/71.7_T2T_ViT_7.pth.tar)  |
| T2T-ViT-10 |    Performer    |   75.2   |  5.9M  | 1.5G | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/75.2_T2T_ViT_10.pth.tar) |
| T2T-ViT-12 |    Performer    |   76.5   |  6.9M  | 1.8G | [here](https://github.com/yitu-opensource/T2T-ViT/releases/download/main/76.5_T2T_ViT_12.pth.tar) |
