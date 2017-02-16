#!/usr/local/bin/python3.6
import argparse
import os
import random as r
import specify_rect as SR
import label_image as LI
import pickle

NUM_TRAIN = 3

possible_commands = {"full": "run all steps, from manual labeling through classifying all remaining examples", 
                     "manual-label": "run the manual labeling step", "train":"run the training step, requires manual labeling to be done first", 
                     "x-validate": "run leave-one-out cross-validation on manually-labeled samples", 
                     "label-blank":"use the trained model to label all remaining examples. No promises about the accuracy"}


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Run box_classifier', epilog = "Possible commands: \n"+"\n".join([" - "+i[0] + ": " + i[1] for i in possible_commands.items()]))

parser.add_argument('-c', '--command', required=True, type=str, help="the function to run. One of these:" + str(possible_commands.keys()), action='store')

parser.add_argument('-i', '--images_dir', required=True, type=str, help="the directory containing only image files to be used for this classification", action='store')

parser.add_argument('-o', '--output_dir', required=True, type=str, help="the directory to store models, training labels, and results", action='store')

args = parser.parse_args()

class ClassificationProject:
    def __init__(self, filename, images_dir, num_train, bbox=None, model=None, manual_labels = dict(), result_labels = dict()):
        self.filename = filename
        self.images_dir = images_dir
        self.num_train = num_train
        self.bbox = bbox
        self.model = model
        self.manual_labels = manual_labels
        self.result_labels = result_labels

    @classmethod
    def from_file(cls, filename):
        ans = pickle.load(open(filename))
        assert(isinstance(ans, cls))
        return ans

    def save(self):
        pickle.dump(self, open(self.filename, 'wb'))
    
    def get_bbox(self):
        dir_entries = list(os.scandir(self.images_dir))
        assert(len(dir_entries) >= self.num_train)
        assert(all([x.is_file() for x in dir_entries]))
        box_selecting_image = r.choice(dir_entries)
        self.bbox = SR.select_rectangle(box_selecting_image.path)
        print ("Bounding box defined: {}".format(self.bbox))

    def manually_label(self):
        self.manual_labels = dict()
        self.get_bbox()
        dir_entries = list(os.scandir(self.images_dir))
        assert(len(dir_entries) >= self.num_train)
        assert(all([x.is_file() for x in dir_entries]))

        for i in r.sample(dir_entries, self.num_train):
            label = LI.manually_label(i.path, self.bbox)
            self.manual_labels[i.name] = label

    def __str__(self):
        return "ClassificationProject:\nbbox:{}\nmanual labels:{}".format(self.bbox, len(self.manual_labels))

if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.command == "full":
        pass
    elif args.command == "manual-label":
        project = ClassificationProject(args.output_dir+"/project.pkl", args.images_dir, NUM_TRAIN)
        project.manually_label()
        project.save()
        print(project)
        
    elif args.command == "train":
        pass
    elif args.command == "x-validate":
        pass
    elif args.command == "label-blank":
        pass
    else:
        print ("no, just no")

