# Torchserve only
This example uses the MNIST dataset and the model used to classify the digits can be found here:
```https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/mnist_cnn.pt```

In that same repo is an example how to serve this model using a cutomized handler based on the ImageClassifier. I used the BaseHandler to get familiar with. Since our model is no image classification we need to get started with the BaseHandler. The handler script is called ```mnist_handler_base.py``` and I used this post to write it: https://towardsdatascience.com/deploy-models-and-create-custom-handlers-in-torchserve-fc2d048fbe91

The script for the base handler can be found here: base handler: https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py

Create the .mar-file:
```torch-model-archiver --model-name mnist --version 2.0 --model-file mnist.py --serialized-file mnist_cnn.pt --handler mnist_handler_base.py --force```

```mv mnist.mar model-store/```

Run the docker image:
```
docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store  pytorch/torchserve:latest torchserve --start --model-store model-store --models mnist=mnist.mar
```

Check available models:

```curl http://127.0.0.1:8081/models/```, output
```
{
  "models": [
    {
      "modelName": "mnist",
      "modelUrl": "mnist.mar"
    }
  ]
}

```

Make predictions (test images for mnist can be found here: https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/test_data):
```curl http://127.0.0.1:8080/predictions/mnist -T 3.png```, output  ```3```

# Web app

* in folder ```deployment``` create a Dockerfile from ```pytorch/torchserve:latest```, which only copies the ```.mar``` file to the ```model-store``` folder. (Note: change latest to other version)
    * build the image: ```build -t torchserve-mar:v1```
* subfolder ```app```
    * content: file ```app.py``` and subfolders ```templates``` and ```static```
    * ```templates``` contains html content of app
    * ```static``` is for saving the uploaded images
    * ```app.py``` is script that creates the app to make predictions
    * in folder ```app``` create Dockerfile to run the app
    * build the image ```docker-build -t app:v1```
* in folder ```deployment``` use ```docker-compose.yaml``` to run both services
* start services with ```docker-compose up```
* in browser go to ```localhost:9696```

