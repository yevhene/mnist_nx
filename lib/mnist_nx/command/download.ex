defmodule MnistNx.Command.Download do
  @source_url Application.get_env(:mnist_nx, :source_url)

  @data_dir File.cwd!() |> Path.join("data")

  def run(_) do
    IO.puts("Downloading data from #{@source_url}")

    File.mkdir(@data_dir)

    HTTPoison.start()

    source_files() |> Enum.map(&download_file/1)

    IO.puts("Downloading successfuly finished")
  end

  defp download_file(file) do
    IO.puts("Downloading #{file}")

    %HTTPoison.Response{body: body} =
      HTTPoison.get!(URI.merge(@source_url, file))

    @data_dir |> Path.join(file) |> File.write!(body)
  end

  defp source_files do
    Tuple.to_list(Application.get_env(:mnist_nx, :train)) ++
      Tuple.to_list(Application.get_env(:mnist_nx, :test))
  end
end
