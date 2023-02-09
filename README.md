# Semantic-Segmentation-Unet
Semantic segmentation using Unet on histopathology images<br>
<br>
In this project I explored various image segmentation techniques. I used augmentation to increase the number of training samples and experimented with 3 models and compared their performance on test dataset. For the future scope of this project, I can augment more images, maybe around 200 per image and train the same models to improve the model performance.<br>
U-net base model performed best among the three with dice score of 0.71. There is one challenge that there are occlusion effect segmenting nuclei because of the low dice score labelling each pixel. Also, if the nuclei are from different organs some modifications can be made to also label each nucleus and the organ. <br>


### Input
![image](https://user-images.githubusercontent.com/55094650/217683592-c96e6fae-ee50-48ba-a993-cbd47600d128.png)
### Ground truth
![image](https://user-images.githubusercontent.com/55094650/217683605-433d477a-819a-4d7f-b68c-5fdaf8706112.png)
### Prediction
![image](https://user-images.githubusercontent.com/55094650/217683614-1d74dda9-94c1-4919-abff-0f60fa7cb13c.png)
