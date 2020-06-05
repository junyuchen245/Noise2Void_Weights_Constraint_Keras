# Noise2Void Weights Constraint in Keras/TensorFlow

The paper, <a href="https://ieeexplore.ieee.org/abstract/document/8954066">A. Krull, et al., "Noise2Void - Learning Denoising From Single Noisy Images," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).</a>, proposed a "blind-spot" kernel for image denoising, where the "blind-spot" kernel means a receoptive field excludes the center pixel. This prevents a neural network to learn the identity in a self-supervised training scheme.

The original paper mentioned that implementing a "blind-spot" kernel is not trivial, so instead the authors used a so called masking scheme (details can be found in the paper). However, to the best of my knowledge, this kernel can be implemented using a kernel constraint that sets the center weight to be 0.

Here is an example result:

<img src="https://github.com/junyuchen245/Noise2Void_Weights_Constraint_Keras/blob/master/sample_result.png" width="600"/>

### <a href="https://junyuchen245.github.io"> About Myself</a>
