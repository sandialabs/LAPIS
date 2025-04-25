dog_preprocessed.txt contains the values of a downscaled and preprocessed
image from wikimedia: https://commons.wikimedia.org/wiki/File:YellowLabradorLooking.jpg

For the benchmark, this image is broadcast across the batch dimension (8).

Running:

  python resnet18.py

will download the pre-trained Resnet18 model from torchvision's model collection,
and use LAPIS to compile it to C++. This will produce the file resnet.hpp.
