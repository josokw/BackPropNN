# Back Propagation Neural Network

The goal of any **supervised learning algorithm** is to find a function that best maps a set of inputs to their correct output. The motivation for backpropagation is to train a multi-layered neural network such that it can learn the appropriate internal representations to allow it to learn any arbitrary mapping of input to output. See [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation).

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2cd688b1e3984f63b00fdee04e7dac4b)](https://www.codacy.com/project/josokw/BackPropNN/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=josokw/BackPropNN&amp;utm_campaign=Badge_Grade_Dashboard)
[![CodeFactor](https://www.codefactor.io/repository/github/josokw/backpropnn/badge)](https://www.codefactor.io/repository/github/josokw/backpropnn)

Source: David Miller C++ code example.
Goal: refactoring example code to modern C++.

- Activation function: non-linear **Hyperbolic Tangent**, zero centered making it easier to model inputs that have strongly negative, neutral, and strongly positive values. [Seven types of activation functions](https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/).

## Compiling

The C++ code is compilable by **g++ v9.3.0**.

Goto the *src* directory and type:

```bash
make
```

## Running, training a BackProp NN

Run the code for training a backprop NN:

```bash
./backpropnn ../data/XOR/txor.txt 
```