# 🚦 Classification Using IBM CV Studio and CNN

This repository contains a step-by-step guide to building and deploying a classifier using IBM CV Studio and Convolutional Neural Networks (CNN) with PyTorch.

### 1. Obtain IBM Cloud Feature Code and Activate Trial Account

If you have already applied your IBM Cloud feature code in another course/lab to create an IBM Cloud account, please skip this item, as the code can only be used once.  

> [!IMPORTANT]
> While signing up for an IBM Cloud account, or at a later stage, you may be prompted to enter a credit card. Please note that access to paid services or a credit card is **NOT** required to complete any mandatory labs/projects/assignments here. Those who have a Feature Code can enable trial access to IBM Cloud for a limited duration. Therefore, we strongly recommend that you **DO NOT** enter a credit card to complete the required course components.

> [!NOTE]
> You can get a Feature Code for your academic institute (if it's included) from [Cloud for Education](https://www.ibm.com/cloud/education) or from the course [Introduction to Computer Vision and Image Processing](https://www.coursera.org/learn/introduction-computer-vision-watson-opencv/home/welcome).

> [!WARNING]
> If you still choose to enter your Credit Card to use IBM Cloud services, you will be liable to pay any resulting charges billed to your credit card.

### 2. Set Up Computer Vision Learning Environment

When building Computer Vision applications, we deal with a lot of images, videos, and other assets that require storage. Image processing also requires significant computing power. The ultimate goal of every computer vision project is to have it deployed as part of an application, and every application requires infrastructure to run.

To help you with your learning, IBM is providing storage, compute, and application infrastructure on the cloud. To use these resources, you need to have an IBM Cloud account. This account is free of charge and does not require any commitment. If you already have an account on IBM Cloud, go ahead to the next section; otherwise, follow the steps below to create an IBM Cloud account and associate CV Studio with this account.

### Pre-requisites
You will need an IBM Cloud account to do this lab. If you have not created one already, click on this [link](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-CC0100EN-SkillsNetwork/labs/IBMCloud_accountCreation/CreateIBMCloudAccount.md.html) and follow the instructions to create an IBM Cloud account.

## Step 1: Connect CV Studio to Your IBM Cloud Account
1. Go to [CV Studio](https://vision.skills.network/)
2. Click on ```My Projects``` in the top right corner.
3. Click ```Sign into IBM Cloud``` at the top right corner.

<p align="center">
  <img width="1659" alt="Sign into IBM Cloud" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/96962d6a-3e18-4eab-a4e2-5ef46e586897">
</p>

### You are all set!
Congratulations! Your learning environment is all set with access to storage, compute, and application infrastructure. Please remember your IBM Cloud ID and password. You will need them to log in if prompted.

## Task 1: Gather and Upload Your Data

To train your classifier, you will need to download the dataset containing the "Stop" sign images found [here](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/dataset/stop.zip) and the "not Stop" sign images found [here](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-Coursera/dataset/not_stop.zip). They are zipped in a folder.

1. Click on **Launch App**.
2. Sign in with your created credentials if prompted.
3. Select the template for this project.

<p align="center">
  <img width="1659" alt="Project Template" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/49dbd2a8-d841-469e-a302-0819e29d62a5">
</p>

4. `Unzip` the files, then drag and drop the folders into the `Upload` section in CV Studio.
5. Make sure to wait between each folder to ensure that it has loaded properly.
6. Confirm that images have been loaded.
7. In the `annotate` section, you will see that the images are pre-labeled with the folder name.

Proceed to the next task once you have gathered and uploaded your images and verified that your images are pre-labeled in the `Annotation` tab of CV Studio.

## Task 2: Training Your Classifier

Create your own `Train` run:

1. Click on `Train Model` on the side panel.
2. Create a `New training run`, enter a `Name`, choose `Jupyter notebook` as a "Training tool", and select `Convolutional Neural Networks (CNN) with PyTorch`.

<p align="center">
  <img width="1659" alt="Train Model" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/546d77e1-5bab-4c83-8125-eea679e6f649">
</p>

> [!TIP]
> You can run the notebook by clicking `Open Notebook` in the **TRAIN** column. It will open a Skill Network Labs and run the `train-classification-cnn-pytorch.ipynb` after uploading it.

## Task 3: Deploy to Code Engine

1. Click on `Use Model` and `New Application`.

<p align="center">
  <img width="1659" alt="Use Model" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/816babae-a0c1-4102-b8d0-788a0742af50">
</p>

2. When you click `New Application`, enter an app name, choose `Models in this project`, use the saved train model, and choose the `Test-1-click Deploy your Model to Cloud (Code Engine)` and create the application.

<p align="center">
  <img width="1659" alt="New Application" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/dbc68dc1-a006-4b74-9a4f-eef6ad0396b4">
</p>

3. Once you are done, it will start deploying and tell you when it is ready.

<p align="center">
  <img width="1659" alt="Deployment" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/746d9b5c-a417-4144-8005-e68449afc56e">
</p>

4. Once it is ready, you will see the green READY button:

<p align="center">
  <img width="1659" alt="Ready" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/ea5c255b-38cf-4d38-8316-a0bc9da480d2">
</p>

## Task 4: Test Your Classifier

1. Click on **Open Notebook** to open the Jupyter notebook from the Train model phase.

> [!NOTE]
> Make sure that the result of the training model is visible under **Details of Run**.

<p align="center">
  <img width="1659" alt="Details of Run" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/45144418-4cca-4d56-89db-f46b535e7f47">
</p>

2. Add the below-mentioned code by adding a cell to the end of the model training notebook. _Note: It's already added to the `train-classification-cnn-pytorch.ipynb`._

```python
imageNames = ['stop_1.jpeg','stop_2.jpeg','not_stop_1.jpeg']
for imageName in imageNames:
    image = Image.open(imageName)
    transform = composed = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    x = transform(image)
    z = model(x.unsqueeze_(0))
    _, yhat = torch.max(z.data, 1)
    # print(yhat)
    prediction = "Stop"
    if yhat == 1:
        prediction = "Not Stop"
    imshow_(transform(image), imageName + ": Prediction = " + prediction)
```

3. Now, download the test images.
- You will find test images [here](https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/tree/main/test_dataset). _Note: if you are using Firefox, please right-click the link and select Save ```Link As```._

4. Then, upload the images to the Skill Network Labs.

<img width="1659" alt="Screen_Shot_2021-04-27_at_3 58 39_PM" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/1287b0a7-15a7-4334-a3a0-0e7e73ce528d">

5. After uploading the images, Run the code for testing your classifier.
Take a screenshot of the following image after the prediction. You can find how to take a screenshot for Mac [here](https://support.apple.com/en-ca/HT201361) and Windows [here](https://www.cnet.com/how-to/7-ways-you-can-take-screenshots-in-windows-10/#:~:text=To%20capture%20your%20entire%20screen%20and%20automatically%20save%20the%20screenshot,to%20the%20Pictures%20%3E%20Screenshots%20folder.)

<img width="800" alt="Screen_Shot_2021-04-27_at_3 58 39_PM" src="https://github.com/akhilaprabodha/Classification-using-IBM-CV-Studio-and-CNN/assets/107745538/f2105751-2a4b-4f45-997b-275c94371a70">

Your prediction will show at the top.

## 🎉 Congratulations!
You have successfully trained, deployed, and tested your classifier using IBM CV Studio and CNN.
