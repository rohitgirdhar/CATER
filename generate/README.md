# Generating CATER


## Pre-generated 
A pre-generated version of the dataset as used in the paper is provided [here](https://cmu.box.com/s/bs48v6tvq6h6tfmda4oht1cw71dcfpdz).
There are 4 `zip` files in the link, corresponding to the different versions used for experiments in the paper.

1. `max2action.zip`: With only 2 objects moving in each time segment. We use these for evaluating the atomic and compositional recognition tasks (Tasks 1 and 2).
2. `all_actions.zip`: With all objects moving at each time segment. We use these for evaluating the localization task (Task 3).

For both, a version with camera motion is also provided. Each also contains a `list` folder with subfolder for each task setup (`localize`, `actions_order_uniq` for compositional, and `actions_present` for atomic). The `localize` task also has a `4x4` and `8x8` setting which were used for ablating the grid sizes. Each of those subfolders have a `train.txt` and `val.txt` files, which are formatted as `video_id, class(es)`.

If you'd like to regenerate CATER or generate it with some modifications, please read on.


## Requirements
1. All CLEVR requirements (eg, Blender: the code was tested with v2.79b).
2. This code was tested on CentOS 7. However, since not all dependencies were available locally in my machine, I ran it using in a [singularity](https://sylabs.io/docs/) VM. [Here's](https://cmu.box.com/s/krg7ehliaidruxjk21nfxsa0gge2uf2o) the VM spec that was used in the following code. If you need it, download and set the path in `launch.py`.
3. GPU: This code was tested with TITAN-X GPUs, though it should be compatible with most NVIDIA GPUs. By default it will use all the GPUs on the machine.


## Generating videos

Run `python launch.py` to start generating. Please read through the launch script to change any settings, paths etc. The command line options should also be easy to follow from the script. If using singularity, you'll need to set a data mount dir, and store videos w.r.t that path.

## Generating labels

You can use the `gen_train_test.py` script to generate labels for the dataset for each of the tasks. Change the parameters on the top of the file, and run it.
