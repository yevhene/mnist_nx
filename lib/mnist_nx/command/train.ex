defmodule MnistNx.Command.Train do
  alias MnistNx.Dataset

  def run(_) do
    IO.puts("Loading training data")

    %Dataset{images: images, labels: _labels} = MnistNx.Dataset.load(:train)

    IO.puts("#{elem(images.shape, 0)} images is loaded")
  end
end
