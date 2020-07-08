# CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning

<p float="left">
    <img src="https://rohitgirdhar.github.io/CATER/assets/teaser/CLEVR_new_000105.gif" alt="eg1" width="200"/>
    <img src="https://rohitgirdhar.github.io/CATER/assets/teaser/CLEVR_new_000115.gif" alt="eg1" width="200"/>
    <img src="https://rohitgirdhar.github.io/CATER/assets/teaser/CLEVR_new_000081.gif" alt="eg1" width="200"/>
</p>

[[project page](https://rohitgirdhar.github.io/CATER/)] [[paper](https://arxiv.org/abs/1910.04744)]

If this code helps with your work, please cite:

Rohit Girdhar and Deva Ramanan. **CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning.** In International Conference on Learning Representations (ICLR), 2020.

```bibtex
@inproceedings{girdhar2020cater,
    title = {{CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning}},
    author = {Girdhar, Rohit and Ramanan, Deva},
    booktitle = {ICLR},
    year = 2020
}
```

## Dataset

A pre-generated sample of the dataset used in the paper is provided [here](https://cmu.box.com/s/w1baekogh29fgu3zg7gr6k446xdalgf2) (direct download links [here](https://github.com/rohitgirdhar/CATER/blob/master/generate/README.md#direct-links)).
If you'd like to generate a version of the dataset, please follow instructions in [generate](generate).

## Baselines

We provide code and some basic instructions on setting up some of the baselines in [baselines](baselines) folder.

## Acknowledgements

This code was built upon the [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) codebase and various video recognition codebases for baselines (especially [Non-Local](https://github.com/facebookresearch/video-nonlocal-net)). Many thanks to those authors for making their code available!

## License
CATER is Apache 2.0 licensed, as found in the LICENSE file.
