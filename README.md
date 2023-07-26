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

The main file for the training is `train.py` and the majority of classes are put into `/lib`.

### Data

The data is not provided as we use bought data. But once have acquired your own data, we recommend to create a `/data` directory
at the root of this project. Within you will want a subfolder `/training` for the dataset used for training and `/validation`
for the dataset used for a separate validation step.

You can of course organize however you wish, simply adjust the `config.yaml` file accordingly.
The project expects the data to represent each trading session by a separate `.csv` file and be of **exactly the same size**.
`preprocess.py` is designed for our data. You can ignore it if you preprocess your data elsewhere.
If using the provided script, you should adjust the preprocess script to fit your specific needs.

Keep in mind to include the data headers inside the `config.yaml` as all which are not specified will be dropped.

### Config / Hyperparameter

All project parameters and hyperparameter are defined inside the `config.yaml`.
It is intended to make changes simpler and provide an overview, rather than serving actual, project wide setting.
Most of those variables are simply forwarded in the `train.py`.

### Run

Run: `python train.py`

The `train.py` will start and train a model as specified.
It also includes a validation on the trained data as well as validation on a separate dataset.

## Server

To serve the model you need to install [Docker](https://www.docker.com/products/docker-desktop/) first.
To start the server run `docker-compose up --build` from the root directory.
This will start two containers:

- One hosting a python server to handle & process the client requests & model responses
- One serving the model via tensorflow-serving

### Request

You can then send a POST request to `http://localhost:8000/predict` with the required data as a `Json` body.
The required data must be of the same dimension and semantic nature as the model was trained on.
More specifically it consists of the price data, as well as the 'state specific' data (e.g. `balance`, `entryPrice` & `contracts`).

For the current configuration the request might look like this:

```json
{
  "progress": [
    0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.2
  ],
  "open": [0, 25, 0, 12.5, -100, -25, -150, 0, 250, 200],
  "high": [12.5, 50, 25, 12.5, -75, 0, -50, 12.5, 275, 250],
  "low": [-25, 0, -50, 0, -200, -50, -200, 0, 200, 100],
  "close": [25, 0, 12.5, -100, -25, -50, 0, 250, 200, 500],
  "volume": [788, 122, 850, 657, 234, 888, 1453, 456, 654, 453],
  "contracts": 5,
  "entryPrice": 3500,
  "balance": 10000,
  "tick_size": 0.25,
  "tick_value": 12.5
}
```

Note that the state specific data is not in array form. This is because during training,
this data is always one singular state and is only converted to a sequence/array of one value to fullfil the dimensionality.
Though, this depends on the particular training setup and how the state works.
It is generally advised to reuse the same methods & classes to processing & modelling the data for handling the requests as for the training.

### Response

The response will be a json and look like this:

```json
{ "action": "SHORT", "amount": 50 }
```

Possible `action` are defined in the `/lib/constants.py`.

## NinjaTrader

Our brokerage platform of choice is [NinjaTrader](https://ninjatrader.com/). The simulation/playback of data is free as is the local downloadable platform.
To download the platform itself register and download it. After that set up your workspace to your liking. There are a few key functions to note, especially when scripting.

### Simulation / Playback

To work locally we use the `playback` or `simulation` option. You find those under the tab `connections`.
To use the playback you need to download the respective historical / market replay data. Once connected, you require a chart. You can create charts under the tab `new`.
After the chart is done, it will have icons at the very top for further function, such as adding indicators and strategies. This is important as those are the functions for our custom scripts.
Note: After adding a e.g. strategy, you might have to enable it in the right sight of the strategies window.

### Scripting

Custom scripts such as strategies and indicators are written in [NinjaScript](https://ninjatrader.com/support/helpGuides/nt8/NT%20HelpGuide%20English.html?ninjascript.htm).
It's NinjaTrader's custom language based on C#. Though, be aware that it is fairly limited. You can develop right in their own NinjaScript editor window (tab: `new`).
This repository always holds the latest `TradingAgent.cs` file, which you need to copy to `C:\Users\[User]\Documents\NinjaTrader 8\bin\Custom\Strategies`, given the default installation.
To see outputs such as errors or prints, you need to open the `NinjaScript Output` window (tab: `new`).

Hints for development:

- Be aware that changes to the file inside the NinjaScript editor must be copied to the file inside this repository. As of now, it is not possible to reference strategies outside the NinjaTrader directory.
- When working on a script, simply compiling or saving the file won't update the strategy when loaded, you have to remove and add the strategy again.
- Be aware which properties for a strategy must be set in what way so that the script and the referenced methods trigger properly.

### .NET frameworks

To add third party frameworks you need to add a reference to the respective `.dll` file. Right click inside the NinjaScript editor and open "References".
There you can add the reference to the `.dll` file. The `.dll` files we use are inside the `.net/` folder.
Since it can lead to unexpected behavior and errors we recommend to copy the `.dll` files inside the `C:\Users\[User]\Documents\NinjaTrader 8\bin\Custom` and reference it from there.

Here's also a list referencing all required frameworks and libraries:

- [Newtonsoft](https://github.com/JamesNK/Newtonsoft.Json/releases)

Note: As of now, the libraries/frameworks must be .net 4.8 compliant.
