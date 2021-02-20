defmodule MnistNx.MixProject do
  use Mix.Project

  def project do
    [
      app: :mnist_nx,
      version: "0.1.0",
      elixir: "~> 1.11",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ex_cli, "~> 0.1.6"},
      {:httpoison, "~> 1.8"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      {:nx, "~> 0.1.0-dev",
       github: "elixir-nx/nx", sparse: "nx", override: true}
    ]
  end
end
