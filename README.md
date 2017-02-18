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

> ./run.py -c x-validate -i /home/cohend/mollie_ppm -p /home/cohend/mollie_model/persona1.pkl
> ./run.py -c full -i /home/cohend/mollie_ppm -p /home/cohend/mollie_model/persona1.pkl
> ./run.py -c summary -p /home/cohend/mollie_model/persona1.pkl

> ./run.py -c write-csv -o /home/cohend/mollie_model/personeros.csv -p /home/cohend/mollie_model/persona1.pkl /home/cohend/mollie_model/persona2.pkl

