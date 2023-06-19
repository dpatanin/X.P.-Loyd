# X.P. Loyd

This project investigates the patterns behind profitable futures daytrading with implications to further trading applications.

## Installation

Since we use `tensorflow-gpu 2.10.1`, make sure to install the respective versions for:

- [Python](https://www.python.org/downloads/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit-archive)
- [cuDNN](https://developer.nvidia.com/cudnn)

You can find an overview table for your OS [here](https://www.tensorflow.org/install).

### CUDA

When installing CUDA it may happen that you encounter an error that says you already have a newer version of some software installed.
This can be due to several reasons, especially if you have other Nvidia software such as e.g. Geforce Experience installed.
In any case, uninstall manually the named software and attempt the CUDA installation again.

### cuDNN

You need to register in order to download the respective kit. Once downloaded, extract the .zip file and navigate into the `cuda` folder.
Insight it you should find the following folders:

- bin
- include
- lib

Copy these and navigate to the install location of CUDA. By default it should be `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`.
Now paste the copied folders, replacing the existing ones.

Next copy the path to the bin folder, e.g.: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`.
You want to add this to your `Path` environment variable. Though it may be that the installation already added it by default.
Repeat this step for the libnvvp folder.

After this is done you might want to restart your computer, though this might not be necessary.

### Dependencies

Once the above steps are done, from the root directory run:

```console
    pip install -r requirements.txt
```

Tensorflow uses by default the GPU. If you want to manually check whether GPU is detected/utilized you can run:

```console
    print(tf.config.list_physical_devices('GPU')) // To see number of GPUs detected
    print(device_lib.list_local_devices()) // To see full list of devices
```

## Training

Everything necessary to train the AI is put into the `/ai-training` directory.

### Data

Inside you need to create locally a few directories:

- One directory where your models will be saved to (default: `ai-training/models`)
- One directory holding your data (default: `ai-training/data`)
- By default, the data is intended to be split into:
  - Training data (default: `ai-training/data/training`)
  - Validation data (default: `ai-training/data/validation`)
  - Test data (default: `ai-training/data/test`)

You can of course organize however you wish, simply adjust the `config.yaml` file accordingly.
The project expects the data to represent each trading session by a separate `.csv` file and be of **exactly the same size**.
`preprocess.py` is designed for our data. You can ignore it if you preprocess your data elsewhere.
If using the provided script, you should adjust the preprocess script to fit your specific needs.

Keep in mind to include the data headers inside the `config.yaml` as all which are not specified will be dropped.

### Config / Hyperparameter

All project parameters and hyperparameter are defined inside the `ai-training/config.yaml`.
It is intended to make changes simpler and provide an overview, rather than serving actual, project wide setting.
Most of those variables are simply forwarded in the `master.py`.

### Run

Run: `python master.py`

The `master.py` will start and train a model as specified.
It also includes a validation/test block which may lead to inaccurate results if run during the same initiation as the training.
We recommend doing separate runs.

**Important:** Don't forget to set the epsilon right to avoid random behavior during validation.

## Server

To serve the model you need to install [Docker](https://www.docker.com/products/docker-desktop/) first.
To start the server run `docker-compose up --build` from the root directory.
This will start two containers:

- One hosting a python server to handle & process the client requests & model responses
- One serving the model via tensorflow-serving
