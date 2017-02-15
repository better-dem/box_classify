#!/usr/local/bin/python3.6
# pdf_conv.py
# convert pdf files to PPM images using pdftoppm

import argparse
import os
import subprocess
from multiprocessing import Pool, Process, Queue

q = Queue()

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Convert all pdfs in input_dir to ppm images in output_dir')

parser.add_argument('-p', '--page', required=True, type=str, help="which page of the pdf should we work with", action='store')

parser.add_argument('-i', '--input_dir', required=True, type=str, help="the directory containing only pdf files to be converted", action='store')

parser.add_argument('-o', '--output_dir', required=True, type=str, help="the directory to store PPM files created from the pdfs", action='store')


args = parser.parse_args()


def update_progress():
    done_counter = 0
    error_counter = 0
    while True:
        update = q.get()
        if update == "error":
            error_counter += 1
        elif update == "success":
            done_counter += 1
        elif update == "done":
            print ("num errors {}, num converted {}".format(error_counter, done_counter))
            break
        if (done_counter + error_counter)  % 100 == 0:
            print ("num errors {}, num converted {}".format(error_counter, done_counter))
            

def conv(infile, page, outfile):
    try:
        subprocess.check_output(["gs", "-sDEVICE=jpeg", "-dFirstPage="+str(page), "-dLastPage="+str(page), "-o", outfile, infile], timeout=30)
        q.put("success")
    except:
        q.put("error")

if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dir_entries = list(os.scandir(args.input_dir))
    dir_entries = [e for e in dir_entries if e.is_file() and e.name.endswith(".pdf")]
    assert(all([x.is_file() and x.name.endswith(".pdf") for x in dir_entries]))
    print ("Number of pdfs: {}".format(len(dir_entries)))

    p = Process(target=update_progress)
    p.start()
    with Pool(4) as p:
        p.starmap(conv, [(e.path, 1, args.output_dir+"/"+e.name[0:-4]+".jpg") for e in dir_entries])

    q.put("done")
    p.join()

    print("everything done")
        

