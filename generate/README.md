# Generating CATER


## Pre-generated
A pre-generated version of the dataset as used in the paper is provided [here](https://cmu.box.com/s/w1baekogh29fgu3zg7gr6k446xdalgf2).
There are 4 folders at the link, corresponding to the different versions of the data used for experiments in the paper. Each contains a zip file with the videos, metadata, and a folder with the train/val files. The train files are further split into `train_subsetT.txt` (80% of `train.txt`) and `train_subsetV.txt` (20% of `train.txt`) as validation sets that could be used for hyperparameter optimization.

1. `max2action`: With only 2 objects moving in each time segment. We use these for evaluating the atomic and compositional recognition tasks (Tasks 1 and 2).
2. `all_actions`: With all objects moving at each time segment. We use these for evaluating the localization task (Task 3).

For both, a version with camera motion is also provided. Each also contains a `list` folder with subfolder for each task setup (`localize`, `actions_order_uniq` for compositional, and `actions_present` for atomic). The `localize` task also has a `4x4` and `8x8` setting which were used for ablating the grid sizes. Each of those subfolders have a `train.txt` and `val.txt` files, which are formatted as `video_id, class(es)`.

### Direct Links
Here are the direct links to the large ZIP files, which should make it easier to download on headless servers.

- all_actions
   - [videos.zip](https://cmu.box.com/shared/static/97kztikj3zwcgv5zu2by6c1jhlvh1d7x.zip)
   - [scenes.zip](https://cmu.box.com/shared/static/6pfetd8s5k7f6bettrpfv11xiqsgou39.zip)
   - [lists.zip](https://cmu.box.com/shared/static/jr2cc6zomqb01hmzvjztpuip3h0sz8vz.zip)
- all_actions_cameramotion
   - [videos.zip](https://cmu.box.com/shared/static/fevyo9fyzeb6vm418yaqjp3gi7tl9azb.zip)
   - [scenes.zip](https://cmu.box.com/shared/static/4u3v9rgusav887i9jfsesfa32yhoo104.zip)
   - [lists.zip](https://cmu.box.com/shared/static/zua3a0afxtuxrkh3jnnkduwkodv816k1.zip)
- max2action
   - [videos.zip](https://cmu.box.com/shared/static/jgbch9enrcfvxtwkrqsdbitwvuwnopl0.zip)
   - [scenes.zip](https://cmu.box.com/shared/static/922x4qs3feynstjj42muecrlch1o7pmv.zip)
   - [lists.zip](https://cmu.box.com/shared/static/7svgta3kqat1jhe9kp0zuptt3vrvarzw.zip)
- max2action_cameramotion
   - [videos.zip](https://cmu.box.com/shared/static/yvhx9p5haip5abzh9i2fofssjpq34zwz.zip)
   - [scenes.zip](https://cmu.box.com/shared/static/zfau8j1e6n7ylobf0g1d2wjdgdu86j2e.zip)
   - [lists.zip](https://cmu.box.com/shared/static/i9kexj33if00t338esnw93uzm5f6sfar.zip)


If you'd like to regenerate CATER or generate it with some modifications, please read on.


## Requirements
1. All CLEVR requirements (eg, Blender: the code was tested with v2.79b).
2. This code was tested on CentOS 7. However, since not all dependencies were available locally in my machine, I ran it using in a [singularity](https://sylabs.io/docs/) VM. [Here's](https://cmu.box.com/shared/static/krg7ehliaidruxjk21nfxsa0gge2uf2o.img) the VM spec that was used in the following code. If you need it, download and set the path in `launch.py`.
3. GPU: This code was tested with TITAN-X GPUs, though it should be compatible with most NVIDIA GPUs. By default it will use all the GPUs on the machine.


## Generating videos

Run `python launch.py` to start generating. Please read through the launch script to change any settings, paths etc. The command line options should also be easy to follow from the script. If using singularity, you'll need to set a data mount dir, and store videos w.r.t that path.

## Generating labels

You can use the `gen_train_test.py` script to generate labels for the dataset for each of the tasks. Change the parameters on the top of the file, and run it.
