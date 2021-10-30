# neuralODE

This is an application of a [Neural ODE](https://arxiv.org/pdf/1806.07366.pdf) to classifying a music into the categories of jazz, rock, and classical music. The way the algorithm works is by taking the [Mel Spectrogram](https://developer.apple.com/documentation/accelerate/computing_the_mel_spectrum_using_linear_algebra) of the audio, thus converting it into an image file. We convolve the image with some kernels, then run the Neural ODE using the convolved images as inputs.

I achieved 97.5 test accuracy by messing with the hyperparameters. 
