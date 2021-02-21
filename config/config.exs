use Mix.Config

config :mnist_nx,
  source_url: "http://yann.lecun.com/exdb/mnist/",
  train: {"train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"},
  test: {"t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"}

config :exla, :clients,
  default: [platform: :host],
  cuda: [platform: :cuda]
