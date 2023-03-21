
# Here is my implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.  

In addition to the MoCo framework, I implemented a decoder network to recover images from the latent space.  With this architecture, one can create a latent space that is robust to various augmentations (thanks to MoCo) and contains information that lets the decoder network recover the original image.  I used this network on DES galaxy thumbnails available publicly (https://des.ncsa.illinois.edu/desaccess/docs/apps.html)

Here's a sample of the DES galaxies that the network was trained on
<p align="center">
  <img src="https://github.com/grantmerz/moco/blob/main/example_images.png" width="500">
</p>


If we train an autoencoder network, we can generate a latent space in which each galaxy image can be represented. This latent space contains a bunch of information about the original image.  It's the decoder network's job to use this latent space to reconstruct the information.  In theory, we want the latent space to be robust against various augmentations.  For instance, a flipped image should pretty much have the same latent space values as its unflipped original. In other words, a flipped image should be the most similar to its unflipped original.  We can measure similarity by computing the distance between latent space values of each galaxy pair and finding the galaxy with the smallest distance.  First, we can look at a baseline convolutional autoencoder without MoCo.  After training, we compute latent space vectors and similarities.
<p align="center">
  <img src="https://github.com/grantmerz/moco/blob/main/sim_placement_nomoco.png" width="700">
</p>

This shows how similar the flipped galaxies are compared to their original counterpart.  Flip1 = xaxis reflection, Flip2=yaxis reflection and Flip3=both axes.  Ideally, all flipped galaxies should be first place in similarity. However, some galaxies are very, very dissimilar to their original versions! In astronomy, there is no preferred orientation of galaxies, so a representation of a galaxy should be robust to spatial augmentations.  Let's see if we can improve this with MoCo.  We use the same AE architecture, but add the momentum contrastive loss to encourage the network to encode augmentated version of the same galaxy into similar latent space values.  Check out those results.

<p align="center">
  <img src="https://github.com/grantmerz/moco/blob/main/sim_placement_moco.png" width="700">
</p>

Every flipped galaxy is correctly recgonized as being the most similar to its original version!  Our latent space is robust to spatial rotations!  We could employ more augmentations to tailor how we want our latent space to be robust.


We can also check the decoder to see if we can recover information back to the pixel space.  It turns out that the best setup involves a downweighting of the reconstruction loss at regions of empty space. This acts as a regularizer in order to recover information across a variety of images. For terrestrial datasets with little empty space, this might not be needed.  Or you could pick certain regions to mask so the network prioritizes different parts of an image
<p align="center">
  <img src="https://github.com/grantmerz/moco/blob/main/raw_recon_comp_128.png" width="700">
</p>

Not everything is recovered, and some color information is scrambled. With a larger sample, its likely to improve.

## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

This is a PyTorch implementation of the [MoCo paper](https://arxiv.org/abs/1911.05722):
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
It also includes the implementation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

This repo aims to be minimal modifications on that code. Check the modifications by:
```
diff main_moco.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
diff main_lincls.py <(curl https://raw.githubusercontent.com/pytorch/examples/master/imagenet/main.py)
```


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```

Linear classification results on ImageNet using this repo with 8 NVIDIA V100 GPUs :
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">pre-train<br/>time</th>
<th valign="bottom">MoCo v1<br/>top-1 acc.</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">53 hours</td>
<td align="center">60.8&plusmn;0.2</td>
<td align="center">67.5&plusmn;0.1</td>
</tr>
</tbody></table>

Here we run 5 trials (of pre-training and linear classification) and report mean&plusmn;std: the 5 results of MoCo v1 are {60.6, 60.6, 60.7, 60.9, 61.1}, and of MoCo v2 are {67.7, 67.6, 67.4, 67.6, 67.3}.


### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">mlp</th>
<th valign="bottom">aug+</th>
<th valign="bottom">cos</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model</th>
<th valign="bottom">md5</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://arxiv.org/abs/1911.05722">MoCo v1</a></td>
<td align="center">200</td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center">60.6</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>b251726a</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">200</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">67.7</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>59fd9945</tt></td>
</tr>
<tr><td align="left"><a href="https://arxiv.org/abs/2003.04297">MoCo v2</a></td>
<td align="center">800</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">&#x2713</td>
<td align="center">71.1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar">download</a></td>
<td align="center"><tt>a04e12f8</tt></td>
</tr>
</tbody></table>


### Transferring to Object Detection

See [./detection](detection).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### See Also
* [moco.tensorflow](https://github.com/ppwwyyxx/moco.tensorflow): A TensorFlow re-implementation.
* [Colab notebook](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb): CIFAR demo on Colab GPU.
