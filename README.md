"# Mobile-MRCNN" 
Explore the Mask R-CNN model
https://github.com/matterport/Mask_RCNN

Task 1. Retrain the model (Mask_RCNN folder)
Build our first image segmentation model!
I have tried to finetune the detection for cars by training on Kaggle Search Results Carvana Image Masking Challenge dataset. With my current methodology, the program shows improvement in the confidence of detection. 

Task 2. Build a simple AI API (AI API folder)
Now we have a model. 
A simple REST API with flask. 
I built an AI API which asks to upload multiple images and returns a page with images & corresponding segmented images.

Task 3. Modify the network (Mobile_mrcnn folder)
Most Mask R-CNN model uses resnet101 as a backbone, which is a humongous model. According to the original paper, the inference time is lower bounded at 200ms which is very slow.

We have read that MobileNet is extremely fast, while still remarkably accurate. Now, we really want to try it out.

The task is to use MobileNet with Mask R-CNN. 
I have implemented the change in the MRCNN backbone from Resnet101 to Mobilenet.

