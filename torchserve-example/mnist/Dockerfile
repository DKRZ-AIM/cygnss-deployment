FROM pytorch/torchserve:latest

COPY ["./model-store", "./model-store"]

CMD ["torchserve", "--start", "--model-store", "model-store", "--models", "mnist=mnist.mar"]
