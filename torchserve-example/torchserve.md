# Deploy a model using torchserve
## Example: densenet

* Clone the git repository: ```git clone https://github.com/pytorch/serve.git``` and install necessary dependencies:
```
cd serve

pip install .

cd model-archiver

pip install .
```

* Go back to folder ```serve``` (```cd ..```)
* Download the model: wget https://download.pytorch.org/models/densenet161-8d451a50.pth
* Create an archive file ```.mar```
```
torch-model-archiver --model-name densenet161 \  #  Encapsulated model name       
                       --version 1.0 \  #  Define version number (optional)
	                   --model-file examples/image_classifier/densenet_161/model.py \  #  Loading status dictionary  Python  Script （ Match the tensor to the layer ）
		                --serialized-file densenet161-8d451a50.pth \   #  Address of model file 
                       --extra-files examples/image_classifier/index_to_name.json \ #  Other documents 
                      --handler image_classifier   #  Invoke excuse
```

* Create directory ```mkdir model-store```
* Save ```.mar``` -file to this folder ```mv densenet161.mar model-store/```
* More general: 
  ```
  torch-model-archiver --model-name <your_model_name> --version 1.0 --model-file <your_model_file>.py --serialized-file <your_model_name>.pth --handler <default_handler> --extra-files ./index_to_name.json
  mkdir model-store
  mv <your_model_name>.mar model-store/
  ```

* Pull docker image ```docker pull pytorch/torchserve:latest```
* Run docker container:
  ```
  docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples pytorch/torchserve:latest torchserve --start --model-store model-store --models densenet161=densenet161.mar
  ```

* (In new tab) view available models: ```curl http://localhost:8081/models```
  ```
  {
    "models": [
      {
        "modelName": "densenet161",
        "modelUrl": "densenet161.mar"
      }
    ]
  }
  ```
* View deployed model ```curl http://localhost:8081/models/densenet161```
```
[
  {
    "modelName": "densenet161",
    "modelVersion": "1.0",
    "modelUrl": "densenet161.mar",
    "runtime": "python",
    "minWorkers": 8,
    "maxWorkers": 8,
    "batchSize": 1,
    "maxBatchDelay": 100,
    "loadedAtStartup": true,
    "workers": [
      {
        "id": "9000",
        "startTime": "2022-06-21T07:03:48.720Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 47,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9001",
        "startTime": "2022-06-21T07:03:48.722Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 48,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9002",
        "startTime": "2022-06-21T07:03:48.722Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 46,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9003",
        "startTime": "2022-06-21T07:03:48.722Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 50,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9004",
        "startTime": "2022-06-21T07:03:48.722Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 49,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9005",
        "startTime": "2022-06-21T07:03:48.722Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 44,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9006",
        "startTime": "2022-06-21T07:03:48.723Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 43,
        "gpu": false,
        "gpuUsage": "N/A"
      },
      {
        "id": "9007",
        "startTime": "2022-06-21T07:03:48.723Z",
        "status": "READY",
        "memoryUsage": 0,
        "pid": 45,
        "gpu": false,
        "gpuUsage": "N/A"
      }
    ]
  }
]
```
* Download an image: ```curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg```
* Make predictions: ```curl http://localhost:8080/predictions/densenet161 -T ../torchserve/inputs/kitten.jpg```
Output:
```
{
  "tabby": 0.46661895513534546,
  "tiger_cat": 0.46449047327041626,
  "Egyptian_cat": 0.06614057719707489,
  "lynx": 0.0012924439506605268,
  "plastic_bag": 0.0002290973934577778
}
```

* Use ```torchserve --stop``` to stop torchserve

* Note: it should also possible to directly create the ```.mar``` file in docker (see [2]). I had problems downloading the model in this case.

* Further Readings:
  * [1] https://pytorch.org/serve/use_cases.html
  * [2] https://github.com/pytorch/serve/blob/master/docker/README.md