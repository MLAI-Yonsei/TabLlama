# TabLlama

## Introduction

In an increasingly complex world, accurate prediction of multi-variate behaviors across domains like aviation, economics, and healthcare has become essential. Traditional models excel at univariate predictions but struggle with capturing the intricate interdependencies inherent in multi-variate data. This paper proposes a hierarchical transformer model designed for multi-variate time series prediction, specifically focused on predicting vehicle behavior. Leveraging the self-attention mechanism, the model captures both long-range and detailed temporal relationships in multi-variate scenarios, while the hierarchical structure facilitates the integration of variable-level information from tabular data. Our approach incorporates novel data augmentation techniques such as noise injection, sequence warping, and swapping to improve model robustness under varied real-world conditions. Experimental results demonstrate that the proposed transformer-based model significantly outperforms conventional baselines, particularly in handling complex multi-variate data. Data augmentation consistently improved prediction performance across multiple metrics, underscoring the model's capability to generalize better under sparse and noisy conditions. The model's superior performance in prediction tasks highlights its potential for enhancing predictive capabilities in aviation and other domains requiring sophisticated multi-variate analysis. By integrating advanced transformer architectures with domain-specific data augmentation, this work addresses a critical gap in current predictive modeling, offering a powerful tool for anticipating complex behaviors in dynamic environments.

Python==3.8.0

```bash
# install environment
pip install -r requirements.txt
```

If you want to train new model, you can set train: True in the config ‘.yaml’ file for each task. Other hyperparameters can also be modified.

### Out-Sequence Task

```bash
python main.py --config ./configs/default/config_default.yaml
```

### In-Sequence Task

```bash
python main.py --config ./configs/indist/config_indist.yaml
```

### Out-Sequence with swapped test dataset Task

```bash
python main.py --config ./configs/testswap/config_testswap.yaml 
```

### In-Sequence with swapped test dataset Task

```bash
python main.py --config ./configs/indist_testswap/config_indist_testswap.yaml
```

### Autoregressive Inference with 1,2,3,4 initial sequences

```bash
bash infers.sh
```

### Reporting Metrics of each Intent

```bash
bash intent_result.sh
```

## Configuration File Explanation

The configuration file (`config.yaml`) defines the settings for training and evaluating the model. Below is a detailed explanation of each parameter:

- **model\_type**: Specifies the type of model to use. Options include `lstm`, `rnn`, `vanillatf`, `tabbert`, `tabgpt2`, `tabllama`. The default is `tabllama`.
- **hidden\_size**: The size of the hidden layers. Default is `4`.
- **num\_layers**: The number of layers in the model. Default is `2`.
- **seq\_num\_layers**: Number of layers in the sequential modeling component. Default is `2`.
- **num\_heads**: Number of attention heads in the transformer. Default is `4`.
- **seq\_num\_heads**: Number of heads for the sequence modeling. Default is `16`.
- **intermediate\_size**: Size of the intermediate layer for transformers. Default is `2400`.
- **num\_epochs**: Number of training epochs. Default is `100`.
- **learning\_rate**: Learning rate for optimization. Default is `0.0001`.
- **file\_path**: Path to the data file. Default is `"path/to/your/data.xml"`.
- **batch\_size**: Number of samples per batch. Default is `32`.
- **col\_dim**: Dimensionality of each column in the input data. Default is `5`.
- **use\_augment\_warp**: Boolean indicating whether to use sequence warping for data augmentation. Default is `False`.
- **use\_random\_noise**: Boolean indicating whether to add random noise for data augmentation. Default is `False`.
- **use\_augment\_swap**: Boolean indicating whether to use feature swapping for data augmentation. Default is `False`.
- **swap\_prob**: Probability of swapping features during augmentation. Default is `0.2`.
- **noise\_mean**: Mean of the noise to be added. Default is `0`.
- **noise\_std**: Standard deviation of the noise. Default is `0.1`.
- **warp\_limit**: Maximum limit for warping sequences. Default is `20`.
- **seed**: Random seed for reproducibility. Default is `0`.
- **patience**: Number of epochs to wait for improvement before stopping early. Default is `10`.
- **min\_delta**: Minimum change to qualify as an improvement. Default is `0.001`.
- **debug**: Boolean flag to enable debug mode. Default is `False`.
- **train**: Indicates if the model should be trained. Default is `True`.
- **infer**: Indicates if inference should be performed. Default is `False`.
- **test**: Indicates if testing should be performed. Default is `False`.
- **in\_dist**: Boolean flag indicating if the data is in-distribution. Default is `False`.
- **test\_swap**: Boolean flag indicating if the test dataset should be swapped. Default is `False`.
- **init\_seq**: Length of the initial sequence for inference. Default is `20`.
- **load\_path**: Path to load the pre-trained model from. Default is `"best_model.pth"`.
- **save\_path**: Path to save the trained model. Default is `"best_model.pth"`.
- **device**: Device to use for computation (`cuda` or `cpu`). Default is `"cuda"`.



## Scenario Generation with `scenario_generator.py`

To generate scenario data using the script `scenario_generator.py`, you can run the script with the following command:

```bash
python scenario_generator.py --time_start 10 --time_end 30 --time_step 1 --n_repeat 200 --scenarios_per_combination 5 --poisson_model exponential --poisson_start 30 --poisson_end 10 --output_file ./xml_testdata.xml --verbose
```

This command runs the scenario generator with specific arguments:

- **`--time_start`**: The starting timestep for the scenario generation. This is required.
- **`--time_end`**: The ending timestep for the scenario generation. This is also required.
- **`--time_step`**: The step size between each timestep. Default is `1`.
- **`--n_repeat`**: Number of times to repeat the data generation process. Default is `1`.
- **`--scenarios_per_combination`**: Number of scenarios to generate for each combination of parameters and time steps. Default is `40`.
- **`--poisson_model`**: Specifies the type of Poisson distribution model to use for scenario generation. Options include `linear` or `exponential`. Default is `linear`.
- **`--poisson_start`**: The starting parameter for the Poisson distribution. Default is `30`.
- **`--poisson_end`**: The ending parameter for the Poisson distribution. Default is `10`.
- **`--output_file`**: The path to the output XML file where the generated scenarios will be saved. Default is `./multiple_scenario_data_test.xml`.
- **`--verbose`**: If provided, the script will print detailed logs during execution.

### Explanation of `argparse`

The `argparse` module in Python is used here to provide an easy way to handle command-line arguments. Below are the arguments defined in the script:

- **`argparse.ArgumentParser`**: Creates an argument parser object that is used to collect arguments.
- **`add_argument()`**: This method is used to specify which command-line options the program is expecting.
  - For example, `--time_start` and `--time_end` are required parameters, meaning they must be provided by the user when running the script.
  - Other arguments like `--n_repeat`, `--time_step`, etc., have default values, which means they are optional.
- **`args = parser.parse_args()`**: Parses the command-line arguments and returns them as a namespace. This allows access to the arguments in the script by referring to `args.argument_name`.

Using `argparse` makes the script flexible and easy to use, as users can specify different parameters for scenario generation based on their needs.


## Analaysis files

To see the results of Autoregressive Inference and its performance by Intent, you can run the following Jupyter notebook file:

- **Plotting Inference Results**: `./src/plot.ipynb` You can run the jupyter notebook file in full to visually see the results of the model's Autoregressive Inference and the performance for each Intent..

- **Scenario Data Generation**: `./src/scenario_generation.ipynb` You can run the jupyter notebook file in full to see the statistics of the scenario data.


## Source Code Structure and Description

Below is a description of the files and directories in the `src` folder:

#### Directories

- **`configs/`**: Directory containing configuration files used to specify different training or testing setups for the model.
- **`models/`**: Directory that contains the implementation of various model architectures including transformer and other RNN-based models.
- **`training_results/`**: Directory used to store the results of the training processes, such as logs and trained models.

#### Python Files (`.py`)

- **`data_processor.py`**: Script that contains functions for data preprocessing, including normalization and handling missing values.
- **`dataset.py`**: Manages loading and preparing datasets for training, including transformations and batching. Recently updated for improved functionality.
- **`main.py`**: Entry point script for training, evaluating, and running inference with the model. It utilizes the configurations specified in the `configs` folder.
- **`trainer.py`**: Script that handles the entire training loop, including validation, early stopping, and saving checkpoints.
- **`utils.py`**: Utility functions that support various tasks such as metric calculations, logging, and data manipulations used throughout the codebase.

#### Jupyter Notebooks (`.ipynb`)

- **`plot.ipynb`**: Jupyter notebook for visualizing model performance and inference results, helping to understand how well the model is performing across different metrics.
- **`scenario_generation.ipynb`**: Jupyter notebook used to analyse synthetic scenario data for training and testing the model.

#### Shell Scripts (`.sh`)

- **`infers.sh`**: Shell script used to run inference tasks in batch mode.
- **`intent_result.sh`**: Shell script for generating intent-wise evaluation results after model inference.

