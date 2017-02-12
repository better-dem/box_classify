#!/usr/local/bin/python3.6
# pdf_conv.py
# convert pdf files to PPM images using pdftoppm

import argparse
import os
import subprocess

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Convert all pdfs in input_dir to ppm images in output_dir')

parser.add_argument('-i', '--input_dir', required=True, type=str, help="the directory containing only pdf files to be converted", action='store')

parser.add_argument('-o', '--output_dir', required=True, type=str, help="the directory to store PPM files created from the pdfs", action='store')


args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.makedirs(arg.output_dir)

    dir_entries = list(os.scandir(args.input_dir))
    assert(all([x.is_file() and x.name.endswith(".pdf") for x in dir_entries]))
    print ("Number of pdfs: {}".format(len(dir_entries)))

    num_converted = 0
    num_errors = 0
    for e in dir_entries:
        new_fname = e.name[0:-4]+".ppm"
        try:
            subprocess.check_output(["pdftoppm", e.path, args.output_dir+"/"+new_fname], timeout=30)
        except subprocess.CalledProcessError as e:
            num_errors += 1
        else:
            num_converted += 1

    print("Done converting {} pdf files to ppm files. {} errors encountered.".format(num_converted, num_errors))

        # extract bbox image
        # show interface requesting manual label
        pass
        

