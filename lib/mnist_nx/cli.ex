defmodule MnistNx.CLI do
  use ExCLI.DSL, mix_task: :mnist

  alias MnistNx.Command

  name("mnist")
  description("Mnist")

  command :download do
    aliases([:d])
    description("Download data")

    run context do
      Command.Download.run(context)
    end
  end

  command :train do
    aliases([:t])
    description("Train network")

    run context do
      Command.Train.run(context)
    end
  end
end
