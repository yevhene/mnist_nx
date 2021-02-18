defmodule MnistNx.Dataset do
  defstruct [:images, :labels]

  alias MnistNx.Dataset

  def load(kind) do
    {images_file, labels_file} = Application.get_env(:mnist_nx, kind)
    data_dir = File.cwd!() |> Path.join("data")

    %Dataset{
      images: load_images(Path.join(data_dir, images_file)),
      labels: load_labels(Path.join(data_dir, labels_file))
    }
  end

  defp load_images(path) do
    data = uncompressed_data(path)

    <<
      2051::integer-signed-32-big,
      n_images::integer-signed-32-big,
      n_rows::integer-signed-32-big,
      n_columns::integer-signed-32-big,
      images::binary
    >> = data

    images
    |> Nx.from_binary({:u, 8})
    |> Nx.reshape({n_images, n_rows, n_columns})
  end

  defp load_labels(path) do
    data = uncompressed_data(path)

    <<
      2049::32,
      n_labels::32,
      labels::binary-size(n_labels)
    >> = data

    labels
    |> Nx.from_binary({:u, 8})
  end

  defp uncompressed_data(path) do
    {:ok, compressed} = File.read(path)

    compressed
    |> :zlib.gunzip()
  end
end
