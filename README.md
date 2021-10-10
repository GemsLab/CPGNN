# Graph Neural Networks with Heterophily

Jiong Zhu, Ryan Rossi, Anup Rao, Tung Mai, Nedim Lipka, Nesreen K Ahmed, and Danai Koutra. 2021. *Graph Neural Networks with Heterophily*. To Appear In *Proceedings of the AAAI Conference on Artificial Intelligence*.

[[Paper]](https://arxiv.org/abs/2009.13566)

## Requirements

This repository is built on top of the experimental pipeline of [H2GCN](https://github.com/GemsLab/H2GCN).

### Basic Requirements

- **Python** >= 3.7 (tested on 3.8)
- **signac**: this package utilizes [signac](https://signac.io) to manage experiment data and jobs. signac can be installed with the following command:

  ```bash
  pip install signac==1.1 signac-flow==0.7.1 signac-dashboard
  ```

  Note that the latest version of signac may cause incompatibility.
- **numpy** (tested on 1.18.5)
- **scipy** (tested on 1.5.0)
- **networkx** >= 2.4 (tested on 2.4)
- **scikit-learn** (tested on 0.23.2)

### For `CPGNN`

- **TensorFlow** >= 2.0 (tested on 2.2)

Note that it is possible to use `CPGNN` without `signac` and `scikit-learn` on your own data and experimental framework.

### For baselines

We also include the code for the baseline methods in the repository. These code are mostly the same as the reference implementations provided by the authors, *with our modifications* to add interoperability with our experimental pipeline. For the requirements to run these baselines, please refer to the instructions provided by the original authors of the corresponding code, which could be found in each folder under `/baselines`.

As a general note, TensorFlow 1.15 can be used for all code requiring TensorFlow 1.x; for PyTorch, it is usually fine to use PyTorch 1.6; all code should be able to run under Python >= 3.7. In addition, the [basic requirements](#basic-requirements) must also be met.

## Usage

### Download Datasets

The datasets can be downloaded using the bash scripts provided in `/experiments/cpgnn/scripts`, which also prepare the datasets for use in our experimental framework based on `signac`.

We make use of `signac` to index and manage the datasets: the datasets and experiments are stored in hierarchically organized signac jobs, with the **1st level** storing different graphs, **2nd level** storing different sets of features, and **3rd level** storing different training-validation-test splits. Each level contains its own state points and job documents to differentiate with other jobs.

Use `signac schema` to list all available properties in graph state points; use `signac find` to filter graphs using properties in the state points:

```bash
cd experiments/cpgnn/

# List available properties in graph state points
signac schema

# Find graphs in syn-products with homophily level h=0.1
signac find numNode 10000 h 0.1

# Find real benchmark "Cora"
signac find benchmark true datasetName\.\$regex "cora"
```

`/experiments/cpgnn/utils/signac_tools.py` provides helpful functions to iterate through the data space in Python; more usages of signac can be found in these [documents](https://docs.signac.io/en/latest/).

### Replicate Experiments with `signac`

- To replicate our experiments of each model on specific datasets, use Python scripts in `/experiments/cpgnn`, and the corresponding JSON config files in `/experiments/cpgnn/configs`. For example, to run `CPGNN-MLP-1` and `CPGNN-MLP-2` on our synthetic benchmarks `syn-products`:

  ```bash
  cd experiments/cpgnn/
  python run_hgcn_experiments.py -c configs/syn-products/cpgnn-mlp.json [-i] run [-p PARALLEL_NUM]
  ```

  - Files and results generated in experiments are also stored with signac on top of the hierarchical order introduced above: the **4th level** separates different models, and the **5th level** stores files and results generated in different runs with different parameters of the same model.
  - By default, `stdout` and `stderr` of each model are stored in `terminal_output.log` in the 4th level; use `-i` if you want to see them through your terminal.
  - Use `-p` if you want to run experiments in parallel on multiple graphs (1st level).
  - Baseline models can be run through the following scripts:

    - **GCN, GCN-Cheby**: `run_gcn_experiments.py`
    - **GraphSAGE**: `run_graphsage_experiments.py`
    - **MixHop**: `run_mixhop_experiments.py`
    - **GAT**: `run_gat_experiments.py`
    - **H2GCN, MLP**: `run_hgcn_experiments.py`

- To summarize experiment results of each model on specific datasets to a CSV file, use Python script `/experiments/cpgnn/run_experiments_summarization.py` with the corresponding model name and config file. For example, to summarize `CPGNN` results on our synthetic benchmark `syn-products`:

  ```bash
  cd experiments/cpgnn/
  python run_experiments_summarization.py cpgnn -f configs/syn-products/cpgnn-mlp.json
  ```

- To list all paths of the 3rd level datasets splits used in a experiment (in planetoid format) without running experiments, use the following command:

  ```bash
  cd experiments/cpgnn/
  python run_hgcn_experiments.py -c configs/syn-products/cpgnn-mlp.json --check_paths run
  ```

### Standalone CPGNN Package

Our implementation of CPGNN is stored in the `cpgnn` folder, which can be used as a standalone package on your own data and experimental framework.

Example usages:

- CPGNN-MLP-1

  ```bash
  cd cpgnn
  python run_experiments.py CPGNN planetoid \
    --network_setup M64-D-R-MO-E-BP1 \
    --dataset ind.citeseer \
    --dataset_path ../baselines/gcn/gcn/data/
  ```

- CPGNN-MLP-2

  ```bash
  cd cpgnn
  python run_experiments.py CPGNN planetoid \
    --network_setup --network_setup M64-D-R-MO-E-BP2 \
    --dataset ind.citeseer \
    --dataset_path ../baselines/gcn/gcn/data/
  ```

- CPGNN-Cheby-1

  ```bash
  cd cpgnn
  python run_experiments.py CPGNN planetoid \
    --network_setup GGM64-VS-R-G-GMO-VS-E-BP1 \
    --adj_nhood 0 1 2 \
    --adj_normalize CHEBY \
    --dataset ind.citeseer \
    --dataset_path ../baselines/gcn/gcn/data/
  ```

- CPGNN-Cheby-2

  ```bash
  cd cpgnn
  python run_experiments.py CPGNN planetoid \
    --network_setup GGM64-VS-R-G-GMO-VS-E-BP2 \
    --adj_nhood 0 1 2 \
    --adj_normalize CHEBY \
    --dataset ind.citeseer \
    --dataset_path ../baselines/gcn/gcn/data/
  ```

- Use `--help` for more advanced usages:

  ```bash
  python run_experiments.py CPGNN planetoid --help
  ```

We only support datasets stored in [`planetoid` format](https://github.com/kimiyoung/planetoid). You could also add support to different data formats and models beyond CPGNN and H2GCN by adding your own modules to `/cpgnn/datasets` and `/cpgnn/models`, respectively; check out our code for more details.

## Contact

Please contact Jiong Zhu (jiongzhu@umich.edu) in case you have any questions.

## Citation

Please cite our paper if you make use of this code in your own work:

```bibtex
@inproceedings{zhu2021graph,
  title={Graph Neural Networks with Heterophily},
  author={Zhu, Jiong and Rossi, Ryan A and Rao, Anup and Mai, Tung and Lipka, Nedim and Ahmed, Nesreen K and Koutra, Danai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={11168--11176},
  year={2021}
}
```
