<img width="600" alt="bl_plot" src="docs/assets/abstract.png">

# Behavior Learning (BL)

This is the official GitHub repository for the paper  **"Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data"**(https://arxiv.org/abs/2602.20152). Behavior Learning (BL) is a general-purpose machine learning framework grounded in behavioral science. It unifies predictive performance and intrinsic interpretability within a single modeling paradigm. BL learns explicit optimization structures from data by parameterizing a compositional utility function built from interpretable modular blocks. Each block represents a Utility Maximization Problem (UMP), a foundational framework of decision-making and optimization. BL supports architectures ranging from a single UMP to hierarchical compositions, enabling expressive yet structurally transparent models. Unlike post-hoc explanation methods, BL provides interpretability by design while maintaining strong empirical performance on high-dimensional tasks.

## Installation
blnetwork can be installed via PyPI or directly from GitHub. 

**Pre-requisites:**

```
Python 3.10.9 or higher
pip
```

**For developers**

```
git clone https://github.com/MoonYLiang/Behavior-Learning.git
cd blnetwork
pip install -e .
```

**Installation via github**

```
pip install git+https://github.com/MoonYLiang/Behavior-Learning.git
```

**Installation via PyPI:**
```
pip install blnetwork
```

Requirements

```python
# python==3.10.9
torch>=2.2
numpy>=1.26
pandas>=2.0
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

**Optional: Conda Environment Setup**
For those who prefer using Conda:
```
conda create --name blnetwork-env python=3.10.9
conda activate blnetwork-env
pip install git+https://github.com/MoonYLiang/Behavior-Learning.git  # For GitHub installation
# or
pip install blnetwork  # For PyPI installation
```

## Computation Requirements

BL is implemented in PyTorch and supports both CPU and GPU training.

- Small-scale tabular examples run on a single CPU within a few minutes.
- High-dimensional settings may benefit from GPU acceleration (e.g., NVIDIA L40).

For most tabular tasks, CPU training is sufficient.

## Examples

Start with the notebooks in [`examples/`](./examples/):

- [Example 1: Boston Housing (continuous)](./examples/Example_1_boston_housing.ipynb)
- [Example 2: Breast Cancer (classification)](./examples/Example_2_breast_cancer.ipynb)

**You need to install scikit-learn>=1.3 to run Examples**

## Important Hyperparameters

- `hidden_dims`: **BL backbone architecture** (depth of BL blocks). Similar to MLP hidden sizes, though BL often requires smaller widths.

- `first_act_func`: **U-block activation function** (e.g., `tanh`, `"none"`).  
  Controls the nonlinearity of the utility term.  
  `tanh` introduces  diminishing marginal effects, while `"none"` keeps it linear.

- `second_act_func` / `third_act_func`: **C-/T-block activation functions**.  
  Shape the constraint structures of the learned optimization landscape.

- `constrain_lambda`: **λ positivity constraint**.  
  If enabled, enforces λ > 0.

- `export_cfg`: **model structure export configuration**.  
  Controls exporting a structured `.txt` summary of model parameters. Accepts `df` / `feature_names` to align learned parameters with input variable names for interpretability.

Other training hyperparameters (optimizer, learning rate, batch size, weight decay, early stopping, etc.) can generally be tuned similarly to standard MLPs.

## Advice on hyperparameter tuning
In many cases, BL can achieve comparable (or slightly better) performance than an MLP baseline using roughly one third of the hidden width.

Other hyperparameters can be initialized based on standard MLP tuning, and then refined for the specific task.

## Interpreting the exported model structure

The exported model structure contains the **parameterized form of each BL block**, including the weights associated with the utility (`U`), inequality constraint (`C`), and equality constraint (`T`) components.

For an input vector `z`, each BL unit defines
$U(z) = \sum_i a_i z_i, \quad C(z) = \sum_i b_i z_i, \quad T(z) = \sum_i c_i z_i$

and produces the score $B(z) = \lambda_U U(z) - \lambda_C C(z) - \lambda_T T(z).$

From an optimization perspective, this corresponds to a utility-maximization problem

$$
\max_z U(z) \quad s.t. C(z) \le 0, T(z) = 0.
$$

In the implemented model, the constraints are relaxed into a penalized objective using the learned weights $\lambda_U, \lambda_C, \lambda_T$.

In deeper BL architectures, higher-level blocks take outputs of lower-level blocks as inputs, forming a **hierarchical optimization structure**. The exported parameters therefore allow the symbolic structure of the learned model to be reconstructed directly.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{ma2026behaviorlearningbllearning,
  title={Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data},
  author={Zhenyao Ma and Yue Liang and Dongxu Li},
  year={2026},
  eprint={2602.20152},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.20152},
}
```

## Contact

If you have any questions, please contact yue.liang@student.uni-tuebingen.de
