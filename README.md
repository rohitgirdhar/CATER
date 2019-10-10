# CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning

<p float="left">
    <img src="https://rohitgirdhar.github.io/CATER/assets/teaser/CLEVR_new_000105.gif" alt="eg1" width="200"/>
    <img src="https://rohitgirdhar.github.io/CATER/assets/teaser/CLEVR_new_000115.gif" alt="eg1" width="200"/>
    <img src="https://rohitgirdhar.github.io/CATER/assets/teaser/CLEVR_new_000081.gif" alt="eg1" width="200"/>
</p>

[[project page](https://rohitgirdhar.github.io/CATER/)] [[paper](https://arxiv.org/abs/1910.04744)]

If this code helps with your work, please cite:

R. Girdhar and D. Ramanan. **CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning.** arXiv 2019.

```bibtex
@inproceedings{girdhar2019cater,
    title = {{CATER: A diagnostic dataset for Compositional Actions and TEmporal Reasoning}},
    author = {Girdhar, Rohit and Ramanan, Deva},
    booktitle = {arXiv preprint arXiv:1910.04744},
    year = 2019
}
```

## Dataset

A pre-generated sample of the dataset used in the paper is provided [here](https://cmu.box.com/s/bs48v6tvq6h6tfmda4oht1cw71dcfpdz).
If you'd like to generate a version of the dataset, please follow instructions in [generate](generate).

## Baselines

We provide code and some basic instructions on setting up some of the baselines in [baselines](baselines) folder.

## Acknowledgements

This code was built upon the [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen) codebase and various video recognition codebases for baselines (especially [Non-Local](https://github.com/facebookresearch/video-nonlocal-net)). Many thanks to those authors for making their code available!

## License
CATER is Apache 2.0 licensed, as found in the LICENSE file.
