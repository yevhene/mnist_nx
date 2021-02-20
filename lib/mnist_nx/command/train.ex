defmodule MnistNx.Command.Train do
  alias MnistNx.Dataset
  alias MnistNx.Model

  def run(_) do
    IO.puts("Loading training data")

    %Dataset{images: images, labels: labels} = MnistNx.Dataset.load(:train)

    {n_images, n_rows, n_columns} = images.shape
    {n_labels} = labels.shape

    IO.puts("#{n_images} images and #{n_labels} is loaded")

    x =
      images
      |> Nx.reshape({n_images, n_rows * n_columns}, names: [:batch, :input])
      |> Nx.divide(255.0)
      |> Nx.to_batched_list(30)

    y =
      labels
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> to_one_hot()
      |> Nx.to_batched_list(30)

    train(x, y)
  end

  defp train(x, y) do
    zip = Enum.zip(x, y) |> Enum.with_index()

    for e <- 1..5,
        {{x, y}, b} <- zip,
        reduce: Model.init_params() do
      params ->
        IO.puts("epoch #{e}, batch #{b}")
        Model.update(params, x, y)
    end
  end

  defp to_one_hot(%Nx.Tensor{} = t) do
    t
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
  end
end
