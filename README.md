# X.P. Loyd

Currently this hosts an example project to get familiar with programming an RL AI in general.
It utilizes a RL Deep-Q Learning Algorithm for predicting Stock Market trades.
Here's the [guide].
Be aware that the guide is not technically correct nor uses best practices, thus the differences in code.

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

Next open copy the path to the bin folder, e.g.: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`.
You want to add this to your `Path` environment variable. Though it may be that the installation already added it by default.
Repeat this step for the libnvvp folder.

After this is done you might want to restart your computer, though it might not be necessary.

### Dependencies

Once the above steps are done, from the root directory run:

```console
    pip install -r requirements.txt
```

Tensorflow uses by default the GPU. The script will check for available GPUs but if you want to manually check you can run:

```console
    print(tf.config.list_physical_devices('GPU')) // To see number of GPUs detected
    print(device_lib.list_local_devices()) // To see full list of devices
```

## Run

Run `master.py`

[guide]: https://www.mlq.ai/deep-reinforcement-learning-for-trading-with-tensorflow-2-0/
