<div align="center">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/atharvaaalok/deepfusion/main/assets/logos/Light_TextRight.svg">
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/atharvaaalok/deepfusion/main/assets/logos/Dark_TextRight.svg">
        <img alt="DeepFusion Logo with text below it. Displays the light version in light mode and
        the dark version logo in dark mode." src="https://raw.githubusercontent.com/atharvaaalok/deepfusion/main/assets/logos/Light_TextRight.svg" width="100%">
    </picture>
</div>

<br>

DeepFusion is a highly modular and customizable deep learning framework.

It is designed to provide strong and explicit control over all data, operations and parameters while
maintaining a simple and intuitive code base.


## Table of Contents
- [Table of Contents](#table-of-contents)
- [DeepFusion Framework](#deepfusion-framework)
- [Basic Usage](#basic-usage)
- [Highlights](#highlights)
  - [1. Customizable training](#1-customizable-training)
  - [2. Gradient Checking](#2-gradient-checking)
- [Installation](#installation)
  - [1. Basic Installation](#1-basic-installation)
  - [2. GPU Training](#2-gpu-training)
  - [3. Network Visualization](#3-network-visualization)
  - [4. All Dependencies](#4-all-dependencies)
- [Resources](#resources)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Credits](#credits)


## DeepFusion Framework
In DeepFusion, all networks are composed by combining 3 basic types of `components`:
- `Data`
- `Module`
- `Net`

`Data` objects hold the network activations and `Module` objects perform operations on them. The
`Net` object forms a thin wrapper around the `Data` and `Module` objects and is used to perform
the forward and backward passes.

A simple neural network is shown below, where, ellipses represent `Data` objects and rectangles
represent `Module`.

![Basic Neural Network](https://raw.githubusercontent.com/atharvaaalok/deepfusion/main/assets/readme_assets/Basic_NeuralNetwork.svg)

> Note the alternating sequence of `Data` and `Module`. The scheme is `Data` -> `Module` -> `Data`.
> Red represents nodes with updatable parameters.

Every node (`Data` or `Module`) has a unique *ID* (for eg: z1 or MatMul1) using which it can be
accessed and modified thus providing explicit access and control over all data and parameters.

More details on `Data`, `Module` and `Net` functionalities can be found in their respective readmes
in [deepfusion/components](./deepfusion/components/).

This is the basic idea behind deepfusion and any and all neural networks are created using this
procedure of attaching alternating `Data` and `Module` nodes.


## Basic Usage
As described before, in DeepFusion, all networks are composed by combining 3 basic types of
`components`:
- `Data`
- `Module`
- `Net`

The codebase follows the same intuitive structure:
```
deepfusion
└── components
    ├── net
    ├── data
    └── modules
        ├── activation_functions
        ├── loss_functions
        └── matmul.py
```

To construct a neural network we need to import the `Net`, `Data` and required `Module` objects.
```python
# Import Net, Data and necessary Modules
from deepfusion.components.net import Net
from deepfusion.components.data import Data
from deepfusion.components.modules import MatMul
from deepfusion.components.modules.activation_functions import Relu
from deepfusion.components.modules.loss_functions import MSELoss
```
> The codebase is designed in an intuitive manner. Let's see how we would think about the above
> imports. "Okay, to create a neural network I need components (deepfusion.components). What kind of
> components do we need? Net, Data and Modules (import these). What kind of modules (operations) do
> we need? we need matrix multiplication, an activation function and a loss function (import these).
> That's it!"

To connect `Data` and `Module` objects we need to keep in mind the following 2 things:
- `Data` objects are used to specify the activation *dimensions*.
- `Module` objects require the *inputs* and *output* data objects to be specified.


Now, let's construct the simple network we saw above.
```python
# Basic structure: x -> Matmul -> z1 -> Relu -> a -> Matmul -> z2, + y -> MSE -> loss
x = Data(ID = 'x', shape = (1, 3))

z1 = Data(ID = 'z1', shape = (1, 5))
Matmul1 = MatMul(ID = 'Matmul1', inputs = [x], output = z1)

a = Data(ID = 'a', shape = (1, 5))
ActF = Relu(ID = 'ActF', inputs = [z1], output = a)

z2 = Data(ID = 'z2', shape = (1, 1))
Matmul2 = MatMul(ID = 'Matmul2', inputs = [a], output = z2)

# Add target variable, loss variable and loss function
y = Data('y', shape = (1, 1))
loss = Data('loss', shape = (1, 1))
LossF = MSELoss(ID = 'LossF', inputs = [z2, y], output = loss)

# Initialize the neural network
net = Net(ID = 'Net', root_nodes = [loss])
```
> For `Data` the first dimension is the batch size. This is specified 1 during initialization. Eg:
> a length 3 vector would have shape = (1, 3) and a conv volume (C, H, W) would have shape =
> (1, C, H, W). During training any batch size (B, 3) or (B, C, H, W) can be used, the `Net` object
> takes care of it.

> Module parameter dimensions are inferred from connected data objects.

Examples introducing the basics and all features of the library can be found in the [demo](./demo/)
directory or in other [resources](#resources).

To have a look at the codebase tree have a look at [Codebase Tree](./assets/codebase_tree.txt).


## Highlights
### 1. Customizable training
Let's say we make the simple neural network as before:

![Basic Neural Network](https://raw.githubusercontent.com/atharvaaalok/deepfusion/main/assets/readme_assets/Basic_NeuralNetwork.svg)
And train it. During training only the *red* portions of the network receive updates and are
trained. Therefore, the matrix multiplication modules will be trained.

Let's say we have trained the network and now we want to find the input that optimizes the function
that we have learnt. This also falls under the same forward-backward-update procedure with the
following simple twist:
```python
net.freeze() # Freezes all modules
x.unfreeze() # Unfreezes the input node
```
After this we obtain the following network:

![Basic Neural Network](https://raw.githubusercontent.com/atharvaaalok/deepfusion/main/assets/readme_assets/Basic_NN_unfrozen_input.svg)
Now when we train the network only the input node value will get updates and be trained!

### 2. Gradient Checking
When developing new modules, the implementation of the backward pass can often be tricky and have
subtle bugs. Deepfusion provides a gradient checking utility that can find the derivatives of the
loss function(s) w.r.t. any specified data object (data node or module parameter). Eg:
```python
# Compare analytic and numeric gradients with a step size of 1e-6 for:
# Input node: x
gradient_checker(net, data_obj = x, h = 1e-6)
# Matrix multiplication parameter W
gradient_checker(net, data_obj = Matmul1.W, h = 1e-6)
```

> [!NOTE]
> Other features such as forward and backward pass profiling, multiple loss functions, automated
> training, gpu training etc. can be found in the [demo](./demo/) directory or in other
> [resources](#resources).


## Installation

### 1. Basic Installation  
To install the core part of deepfusion use:
```
$ pip install deepfusion
```

### 2. GPU Training
To use GPU training capabilities you will require [CuPy](https://pypi.org/project/cupy/) which
needs the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). If the CUDA Toolkit is
installed then use:
```
$ pip install deepfusion[gpu]
```

### 3. Network Visualization
For visualizing networks you will require the [Graphviz](https://graphviz.org/download/) software
and the [graphviz](https://pypi.org/project/graphviz/) package. If Graphviz is installed then use:
```
$ pip install deepfusion[visualization]
```
### 4. All Dependencies
If all dependencies are pre-installed use:
```
pip install deepfusion[gpu,visualization]
```

> [!IMPORTANT]
> Make sure to select add to PATH options when downloading dependency softwares.


## Resources
- [DeepFusion documentation]()
- [DeepFusion demo](./demo/)
- [DeepFusion Tutorials]()


## Contribution Guidelines
Contributions for the following are encouraged and greatly appreciated:
- **Code Optimization:** Benchmark your results and show a clear improvement.
- **Visualization:** Currently requires graphviz which is usually a pain to install. Structured
  graph visualization using say matplotlib would be a clear win.
- **More Modules:** Most scope for contribution currently in the following modules: loss_functions,
  pooling, normalizations, RNN modules etc.
- **More Features:** Some ideas include adding multiprocessing, working with pre-trained models from
  other libraries etc.
- **Testing:** Incorporating testing codes.
- **Improving Documentation:** Improving doc-string clarity and including doc tests. Also perhaps
  making a website for API reference.

We'll use [Github issues](https://github.com/atharvaaalok/deepfusion/issues) for tracking pull
requests and bugs.


## License
Distributed under the [MIT License](License).


## Acknowledgements
Theoretical and code ideas inspired from:
- [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/)
- [Coursera: Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
- [Udemy: Python Packaging](https://www.udemy.com/course/python-packaging/?couponCode=LEADERSALE24B)


## Credits
- Logo design by [Ankur Tiwary](https://github.com/ankurTiwxry)