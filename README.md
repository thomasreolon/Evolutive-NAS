[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-No-red.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Generic badge](https://img.shields.io/badge/python-3.7%20|%203.8-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/version-v1.0-cc.svg)](https://shields.io/)

# Neural Architecture Search with Genetic Algorithms

Hi! this is the project for the BIO-inspired Artificial Intelligence course at unitn.

TL;DR :arrow_right: evolve Neural Network architectires using genetic algorithms (pytorch)
___

## Project structure:

    EVOLUTIVE-NAS
    ├── src
    |    ├── fne                      [library for Fast Neural Evolution]
    |    |    ├── evolution             [Genetic Algorithm part]
    |    |    |    ├── crossover        [class that implements crossover between 2 genotypes]
    |    |    |    ├── dataset          [dinamically get subsets of a dataset]
    |    |    │    ├── fitness          [heuristic functions to evaluate NN trainability]
    |    |    │    ├── mutations        [class that implements mutations]
    |    |    |    ├── utils            [script to extract features from the openface's csv]
    |    |    |
    |    |    ├── genotopheno         [Pytorch & dynamic NN]
    |    |    |    ├── cell_operations  [basic connections (bricks for building cells)]
    |    |    |    ├── cells            [small NN that we evolve]
    |    |    │    ├── network          [combine cells: DARTSnetwork & EvaluationNetwork]
    |    |
    |    |
    |    ├── plot_scores            [evaluate the heuristic functions on some NN]
    |    ├── NAS_on_cifar10         [launch GA on cifar10]


## Report 

you can find the pdf (wich has the same content but with images) here: [report.pdf](report.pdf)

___


Given the recent success of deep learning, new fields that try to improve neural networks are emerging, Neural Architecture Search (NAS) is one of them. We propose a system that can autonomously find good NN architectures using a genetic algorithm. To speedup architectures’ evaluations we use heuristic functions that do not require training the NN, and a variation of DARTS.

NAS, NTK, DARTS, GA, neuroevolution

## Introduction


is possible to embed some prior knowledge in the architecture of a neural network (NN), for example Adam Gaier et al. showed that we can build weight agnostic neural networks, which can perform tasks even without tuning their weights. Moreover many breakthroughs in deep learning came from architectural innovations, eg. the use of CNN in Alexnet (2012), skip-connections in Resnet (2016) , self-attention in Transformers (2017) , …

Even if NAS has just lately become a popular topic, some very interesting approaches like NEAT had already been proposed in the past. This approach uses a genetic algorithm (GA) to evolve the architecture and the weights of a NN. More recently a variation called CoDeepNEAT suggested a way to evolve architectures with more parameters (in NEAT you evolve weights and connections, in CoDeepNEAT you evolve layers and connections between them), moreover, they evolve cells, which can be seen as small NN, and divide them into species; once the evolution phase is complete, they build the final architecture by combining these cells. In this project, we use the same cell-based approach.

Even if the literature contains many BIO-inspired algorithms to perform neural architecture search (eg. deepswarm ), the most popular approaches use other techniques such as weights sharing and network morphisms . Among these approaches a very simple ed effective method is DARTS , which can be summarized as follows: we use backpropagation to learn a parameter called alpha, which selects which connection (dense, cnn, max-pooling, ...) we should keep in our final architecture.

One of the biggest issues in using GA for NAS was how to evaluate an architecture. A commonly used fitness function was the accuracy obtained by the NN after some training epochs, but even if this approach produced very good architectures it required enormous quantity of computing power (eg. AmoebanetA used weeks of GPU computing power ). Given some recently discovered properties of NN, some methods to evaluate architectures without training them were proposed , we base our approach upon these methods. In particular, we adopt the heuristic functions to replace the old accuracy fitness function.

Section II presents our method, Section III shows the obtained performances (plot of the fitness and accuracy function) and section IV contains some thoughts on this subject.

## Method & Implementation


we present the most salient elements of the algorithm, then we explain how an iteration/epoch is run.

#### GENOTYPE


The genotype represents the design of a cell (that is: which layers are used, how they are connected and how many parameters they have). To encode the cells we used a similar approach to the one used in DARTS; we preferred this approach over the encoding used in NEAT because there were more code-examples available online (for pytorch).

#### PHENOTYPE


Pytorch was chosen as deep learning framework to build the phenotypes; we took inspiration on how to dynamically build pytorch’s NN from the TENAS project .

#### FITNESS FUNCTIONS

We have three functions to minimize:

-   the Neural Tangent Kernel score (implementation taken from TENAS project with some changes to avoid running out of GPU-RAM)

-   the Linear Region Score (from )

-   the Gradient Correlation score created following the ideas explained in Yannic Kilcher’s videos

You can find more information about these functions in Appendix A.

#### CROSSOVER


We use a simple crossover strategy: given that the genotype is encoded as a string, the \(i-th\) character of the offspring will be randomly selected between the \(i-th\) character of the two parents.

#### MUTATION


The main mutation strategy consists in adding some new channels in the architecture, there are some parameters which decide how to do these mutation and are learned during the iterations (global evolution strategy). There are some other types of mutations, eg. swapping blocks, reduce the number of channels, …

#### SELECTION


Once we have the 3 scores for each offspring, we select the architectures that were not dominated by others. If there are too many offsprings after this selection we further reduce their numbers with the method proposed in . This is a \((\mu+\lambda)\) GA because parents can survive multiple generations.

#### DARTS STEP


It is probable that minimizing these three scores will introduce some bias in the found architectures. To minimize this bias, once every three generation, we use another approach to evaluate/select architectures. Instead of using the three heuristic functions and the Pareto dominance selection, we create some tournaments between the offsprings where, only one offspring per tournament will survive. This tournament is created adopting the DARTS method but instead of selecting connections we select cells designs.

Once the algorithm completes the final epoch, we store the final population on a file. To obtain the final architecture, we concatenate the cells and add some final pooling/dense layers.


## Conclusions


work is built upon the works of Chen et al. and Mellor et al. and tries to explore the potential of applying genetic algorithms to NAS. Potentially there could still be months of work to make improvements, so the results should be taken lightly.

There were 2 main kinds of problems encountered during this project: technical (eg. GPU-memory handling, creating NN dynamically, ...) and practical (at the beginning it was not working very well due to bad choices of possible mutations, search space (skip-connection are a bit bugged as stated in and too)).

A nice property that many NAS implemented projects have is the capability to change the connections (the bricks of the cells, eg. CNN, avgPooling, ...). This system makes no difference and allows you to create your own connections (as long as they offer the same API of a CNN), this could allow building very interesting experiments (eg. introducing attention or other types of layers). Anyway, this capability is also offered by simpler algorithms such as DARTS or TENAS.

There still are margin of improvements: better heuristic functions, improve mutation and crossover, but we are satisfied with our work. Probably in some months some new papers on this topic will be released.

## Heuristic-Based Fitness Functions

#### Neural Tangent Kernel


How To Compute: We take \(N\) samples, and for each of them we do the forward and the backward pass. We then store the gradients in a matrix \(M\), where \(M_{ij}\) is the gradient of the parameter \(j\) for sample \(i\) (a NN usually has many output neurons, so a better description of the gradients would be: “the flattened jacobian matrix of size: (numberofparameters, number of output neurons)” ). We then compute the co-variance between the samples’ gradients as \(C = M \cdot M^T\). Finally we compute the eigenvalues and return as final score the ratio between the biggest and the smallest one.

Intuition: A neural network has a good potential if there is a lot of variance between the gradients generated by different samples. If the ratio between the 2 eigenvalues is small, we don’t have a clear principal component that describe how they are distributed, thus we have a high entropy/variance. More: .

#### Activation Score


Intuition: A neural network would appreciate if the activations of different inputs were different between each other. This score measures these dissimilarities by computing a matrix \(M\) where \(M_{ij}\) is “how many activations have the same score between sample i and sample j”. If \(M\) is a diagonal matrix the determinant is maximized and the activation are very different between each other (\(score = -log(determinant)\)). More: .

#### Third Score


