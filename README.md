## A generative model of the hippocampal formation trained with theta driven local learning rules
This is the code for replicating results in the NeurIPS 2023 paper ["A generative model of the hippocampal formation trained with theta driven local learning rules"](https://openreview.net/pdf?id=yft4JlxsRf)


### Abstract 
Advances in generative models have recently revolutionised machine learning. Meanwhile, in neuroscience, generative models have long been thought fundamen- tal to animal intelligence. Understanding the biological mechanisms that support these processes promises to shed light on the relationship between biological and artificial intelligence. In animals, the hippocampal formation is thought to learn and use a generative model to support its role in spatial and non-spatial memory. Here we introduce a biologically plausible model of the hippocampal formation tantamount to a Helmholtz machine that we apply to a temporal stream of inputs. A novel component of our model is that fast theta-band oscillations (5-10 Hz) gate the direction of information flow throughout the network, training it akin to a high-frequency wake-sleep algorithm. Our model accurately infers the latent state of high-dimensional sensory environments and generates realistic sensory predic- tions. Furthermore, it can learn to path integrate by developing a ring attractor connectivity structure matching previous theoretical proposals and flexibly transfer this structure between environments. Whereas many models trade-off biological plausibility with generality, our model captures a variety of hippocampal cognitive functions under one biologically plausible local learning rule.

### Code break down
We recommend cloning the repo and running locally but you can also run the code in the cloud (they're just a bit slower) using Google Colab and the links provided.
* `figs12.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomGeorge1234/HelmholtzHippocampus/blob/main/figs12.ipynb)
reproduces core results in figures 1 and 2 of the paper 
* `figs34.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomGeorge1234/HelmholtzHippocampus/blob/main/figs34.ipynb)
reproduces core results in figures 3 and 4 of the paper
* `HH_utils.py` contains helper functions for the models
