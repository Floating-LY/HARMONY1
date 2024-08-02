# HARMONY

This repository provides an example code implementation for the paper titled "Hierarchical Multi-Indicator Distribution Forecasting and Bayesian Decision-Making System for Cloud Resource Scaling". The paper introduces a novel system for efficient cloud resource scaling by addressing key challenges in cloud service resource allocation through hierarchical multi-indicator distribution forecasting and Bayesian decision-making.

## Dataset

As an example, we have included the Fisher dataset used in the paper. The dataset comprises workload data from 10 containers over a 30-day period within a Kubernetes framework sourced from [repo](https://github.com/chrisliu1995/Fisher-model/tree/master), representing a modern containerized environment. The dataset has been preprocessed and ready for use.

## Repository Structure

The repository has the following structure:

```
HARMONY/
├── Dataset/Fisher/
├── data/
│   ├── data_factory.py
│   └── data_loader.py
├── experiments/
│   ├── exp_basic.py
│   └── exp_forecasting.py
├── layers/
│   ├── Embed.py
│   ├── SelfAttention_Family.py
│   ├── StandardNorm.py
│   └── Transformer_EncDec.py
├── model/
│   └── harmony.py
├── utils/
│   ├── masking.py
│   ├── metrics.py
│   ├── timefeatures.py
│   └── tools.py
├── Analysis.ipynb
├── Test_All.py
├── requirements.txt
├── result_Fisher.txt
└── run.py
```

This structure separates the codebase into different modules, making it more organized and maintainable. The `data` directory contains scripts for data processing and loading, while the `experiments` directory holds code for running different experiments. The `layers` directory contains various neural network layers used in the model implementation . The `model` directory contains the main `harmony.py` file, which defines the HARMONY model architecture. The `utils` directory includes utility functions for masking, metrics calculation, time feature extraction, and analysis.

## Usage

To run the code, execute the following command:

```
pip install -r requirements.txt
python Test_All.py
```

After running the code, you can analyze the experimental results using the code modules in the `Analysis.ipynb` notebook. Additionally, the model parameters can be adjusted in the `run.py` file.

## Acknowledgements

Our code implementation was influenced by the [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

Thank you for your interest in HARMONY.
