# Panopticon

# Description

Machine Learning pipeline for PLATO data. We use a CNN to analyze simulated data to prepare for real
data analysis in the near future.

The project is currently under development, more features and usage cases are coming.

---

# Usage

To install the required dependencies, use the `requirements.txt` file with the command:

> `pip install -r requirements.txt`

---

## Machine learning

The machine learning portion of the code is accessed via the `main.py` program. A number of flags then dictate what to do.

> `--train`

Starts the training on the dataset.

> `--eval`

Starts evaluating the model using the parameters specified.

## Data handling

The data analysis, visualization and preparation is done through the `data.py` program.

> `--plot [file_name]`

This flags set the program in plotting mode, which is used to generate the visual of the light curves
from the simulations. It can be called using the name(s) of the simulation file(s) to use (*e.g.*
`--plot sim00001_Q1.ftr sim00002_Q1.ftr`) or directly for all available files (`--plot all`).

To simplify the process, the latest run can be accessed by using `--plot last`.

> `--plotall`

Runs the plotting routine on all files available.

> `--analyze [file_name]`

Similar syntax to the `--plot` flag, but for the analysis of a run.

To prepare data for a run, follow these steps

First, open the `param_maker.ipynb` notebook and generate the `Complete_parameters.ftr` file (make sure to have all the `AllParameters` files from the simulations).

Then run the label creator, which will create the ML model targets, and the packaging operation, which will format them into the required formats for the training and testing.
```
python panopticon.py --labels
python panopticon.py --prepare [--augment --filter_inf]
```

The `--filter_inf` (default = `True`) flag removes `NaN` to avoid gradient computation errors, and `--augment` (default = `False`) creates the inverted time series of a given LC using spline fitting.

## Secondary flags

> `--cfg path/to/config.toml`

The `--cfg` flag is used to pass the path to a configuration file. If not specified, the default
file (`./config/default.toml`). For a description on what is in the configuration file,
see the [dedicated section](##configuration-file).

---

# Configuration file

The configuration file is a `toml` file used to pass key information to run the program. Each entry is
described in the file itself, but here is a quick overview. By default the program uses the `default.toml`
file, to pass a custom one, use `--cfg ./path/to/your/file.toml` when calling the program.

> `path_data`: is the path to the raw simulation files

> `path_prepared`: is the path to the prepared data for training

> `path_output`: The path where the output of the training is written

> `seed`: is the seed to use for the splitting of the datasets (used for reproducibility)

> `training_epochs`: maximum number of epochs during training

> `[fraction]`: are the values corresponding to the proportion of training, validation and testing

> `[dataloaders]`: kwargs to pass when creating the torch DataLoaders
(see [docs](https://pytorch.org/docs/stable/data.html))

> `[optimizer]`: the optimizer to use during training
(and its parameters, [docs](https://pytorch.org/docs/stable/optim.html))

> `[scheduler]`: the scheduler to use during training
(and its parameters, [docs](https://pytorch.org/docs/stable/optim.html))

If you are not familiar with `toml`, [here is the documentation](https://toml.io/en/). In essence, it's 
a very practical format to write a few parameters down without having to hard code them in the
source code. It also makes it easy to create multiple sets of parameters if the code runs on various
machines, for various people.

---

# Other commands

The documentation is built with [`pdoc`](https://pdoc.dev/docs/pdoc.html), to generate it,
use the command:

> `pdoc ./src -o ./docs -t ./doc_theme`

To generate the requirements, use `pipreqs`:

> `pipreqs ./src`

and to install them all, use:

> `pip install -r ./requirements.txt`

--

# Authors

Codebase written by Hugo G. Vivien (Laboratoire d'Astrophysique de Marseille),
under the supervision of Magali Deleuil.

---
