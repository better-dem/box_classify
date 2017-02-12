#!/usr/bin/python3
import argparse
import os
import random as r
import specify_rect as SR

NUM_TRAIN = 100

possible_commands = {"full": "run all steps, from manual labeling through classifying all remaining examples", 
                     "manual-label": "run the manual labeling step", "train":"run the training step, requires manual labeling to be done first", 
                     "x-validate": "run leave-one-out cross-validation on manually-labeled samples", 
                     "label-blank":"use the trained model to label all remaining examples. No promises about the accuracy"}


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Run box_classifier', epilog = "Possible commands: \n"+"\n".join([" - "+i[0] + ": " + i[1] for i in possible_commands.items()]))

parser.add_argument('-c', '--command', required=True, type=str, help="the function to run. One of these:" + str(possible_commands.keys()), action='store')

parser.add_argument('-i', '--images_dir', required=True, type=str, help="the directory containing only image files to be used for this classification", action='store')

parser.add_argument('-o', '--output_dir', required=True, type=str, help="the directory to store models, training labels, and results", action='store')


args = parser.parse_args()

if __name__ == "__main__":
    if args.command == "full":
        pass
    elif args.command == "manual-label":
        print ("Manually Labeling Training Data")
        filenames = os.listdir(args.images_dir)
        assert(len(filenames) >= NUM_TRAIN)
        print ("Number of images: {}".format(len(filenames)))
        box_selecting_image = r.choice(filenames)
        bbox = SR.select_rectangle(box_selecting_image)["rect"]
        print ("Bounding box defined: {}".format(bbox))

        for i in r.sample(filenames, NUM_TRAIN):
            # extract bbox image
            # show interface requesting manual label
            pass
            
        
    elif args.command == "train":
        pass
    elif args.command == "x-validate":
        pass
    elif args.command == "label-blank":
        pass
    else:
        print ("no, just no")

