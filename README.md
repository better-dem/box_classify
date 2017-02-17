# box_classify
A computer vision tool to extract boxes from similar images (particularly, scanned paperwork) and classify them.


## Requirements

Python 3.6

sudo pip3.6 install numpy scikit-learn scikit-image

On ubuntu:

> sudo apt-get install python3-tk python3-pil python3-pil.imagetk

And ghostscript
(ex: gs -sDEVICE=jpeg -o /home/cohend/box_classify/pic-%d.jpg ~/mollie/033922.pdf)

## Example Usage

> ./run.py -c x-validate -i /home/cohend/mollie_ppm -o /home/cohend/mollie_model


> ./run.py -c manual-label -i /home/cohend/mollie_ppm -o /home/cohend/mollie_model
