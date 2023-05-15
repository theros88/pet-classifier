# A cats' and dog's breed Azure WebApp Classifier
This is a toy Deep Learning classifier which is trained to label different breeds of cats and dogs.

It shows the full ML pipeline from training to Azure deployment as a web application.

## Training
A pre-trained DenseNet NN is used to fine-tune the final model. The framework which was used for training the model is [fastai](https://www.fast.ai/). Standard fastai augmentations were used (flip, rotate, zoom, warp, lighting transforms etc.) during training. 

The images and the annotations for training were taken from [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). 7393 images were used with 20% of that being the validation set.

The `Training_a_pet_Classifier.ipynb` Colab notebook was used for the training, with a single GPU, that took something less than 20 minutes to train with a fairly good final error rate of 5.7%. The resulting model can be found in the `models` folder.

## WebApplication
The [Dash](https://plotly.com/dash/) framework was used for creating the WebApp as it allows to use pythonic syntax to create the HTML page inside the server code and the outcome is quite simple and easily maintained. These all can be seen in the `app.py` server code.  

For inference from within the server, no GPU is required as the user can only upload the images only one at a time, so the PyTorch library with only CPU support was preffered and results to a much smaller container.

## Testing
To run and test the WebAPP, use:
```
python app.py
```

This will start running the server which can be accessed locally with a web browser at `localhost:8050`. 

In order to make changes of the WebApp at run-time and see the changes, without restarting the server, the following line of the `app.py` should read:
```
app.run_server(host='0.0.0.0',port=8050,debug=True)
```

A sample screenshot of the WebApp follows:


## Containerization
In order to create a container, docker was used with its corresponding `Dockerfile` for creation and final deployment. 

You could create the docker container with the commands:
```
git git@github.com:theros88/pet-classifier.git 
cd pet-classifier
docker -t create -t <YOUR_ACR_NAME>.azurecr.io/pet-class-app:latest .
```
(See **Azure Deployment** below for the ACR NAME. For this particular project, I chose `petclassifier`, so the whole docker tag was `petclassifier.azurecr.io/pet-class-app:latest`) 

In order to test the container locally, you could use:
```
docker run <YOUR_ACR_NAME>.azurecr.io/pet-class-app

```
This will start running the server in the docker container and can be accessed locally with a web browser at `http://172.17.0.2:8050`. 

## Azure Deployment
### Uploading the container
Firstly, you have to create an Azure Container Registry (ACR) on Azure with a selected ACR name of your choice. To be able to upload the container onto this ACR you need to authenticate Azure credentials on your local machine. Use the following command to do that:

```
docker login <YOUR_ACR_NAME>.azurecr.io
```
You will be prompted for a username and password. The username is the name of your registry (in this example the username is `petclassifier`). You can find your password under the access keys of the ACR resource you've created. After authentication,
 you can push the container you have created to ACR by using the following:
```
docker push <YOUR_ACR_NAME>.azurecr.io/pet-class-app:latest
```
You need a machine with access to high-speed internet as this is a rather large file (~2.5GB) and this process will take a while.

### Creating the WebApp
To create a web app on Azure you have to login on the Azure portal, create a new WebAPP resource and finally, link your ACR image to this WebApp resource.

A final step is necessary as by default Azure WebApps expose only `port:80` and our web server runs at `port:8050`. In WebApp's panel under Configuration &rarr; Application Settings, you have to add a setting `WEBSITES_PORT` with value `8050`.

Now start the WebApp and it should be up and running!

The WebApp can be seen running live in [Cat & Dog Breed Recognizer](https://petbreed.azurewebsites.net/).

## Further Developments
- Monitoring the use of the WebApp
- Updating the WebApp with an extra feature (e.g. recognition of multiple pets in an image) and repeat the whole ML life-cycle all over with some degree of automation.