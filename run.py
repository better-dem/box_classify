#!/usr/local/bin/python3.6
import argparse
import os
import random as r
import specify_rect as SR
import label_image as LI
import pickle
from sklearn import svm
from skimage import feature, color
from PIL import Image
import numpy as np

NUM_TRAIN = 100                 # size of training set
SCALE_ALL_IMAGES = True         # scale all images to be the size of the image used to select a bounding box

possible_commands = {"full": "run all steps, from manual labeling through classifying all remaining examples", 
                     "manual-label": "run the manual labeling step", "train":"run the training step, requires manual labeling to be done first", 
                     "x-validate": "run leave-one-out cross-validation on manually-labeled samples", 
                     "label-blank":"use the trained model to label all remaining examples. No promises about the accuracy",
                     "summary":"Load a project from the output directory and summarize it"}


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Run box_classifier', epilog = "Possible commands: \n"+"\n".join([" - "+i[0] + ": " + i[1] for i in possible_commands.items()]))

parser.add_argument('-c', '--command', required=True, type=str, help="the function to run. One of these:" + str(possible_commands.keys()), action='store')

parser.add_argument('-i', '--images_dir', required=True, type=str, help="the directory containing only image files to be used for this classification", action='store')

parser.add_argument('-o', '--output_dir', required=True, type=str, help="the directory to store models, training labels, and results", action='store')

args = parser.parse_args()

class ClassificationProject:
    def __init__(self, filename, images_dir, num_train, image_size=None, bbox=None, model=None, manual_labels = dict(), result_labels = dict()):
        self.filename = filename
        self.images_dir = images_dir
        self.num_train = num_train
        self.image_size = image_size
        self.bbox = bbox
        self.model = model
        self.manual_labels = manual_labels
        self.result_labels = result_labels

    @classmethod
    def from_file(cls, filename):
        try:
            ans = pickle.load(open(filename, 'rb'))
            assert(isinstance(ans, cls))
            return ans
        except:
            return None

    def save(self):
        pickle.dump(self, open(self.filename, 'wb'))
    
    def get_size(self):
        """
        Take the median (by each dimension?) of several image sizes 
        """
        num_images = 15
        dir_entries = list(os.scandir(self.images_dir))
        assert(len(dir_entries) >= num_images)
        assert(all([x.is_file() for x in dir_entries]))
        sizes = []
        for i in r.sample(dir_entries, num_images):
            im = Image.open(i.path)
            sizes.append(im.size)
        s0 = int(np.median([x[0] for x in sizes]))
        s1 = int(np.median([x[1] for x in sizes]))
        self.image_size = (s0, s1)

    def get_bbox(self):
        self.get_size()
        dir_entries = list(os.scandir(self.images_dir))
        assert(all([x.is_file() for x in dir_entries]))
        box_selecting_image = r.choice(dir_entries)
        self.bbox = SR.select_rectangle(box_selecting_image.path, self.image_size)
        print ("Bounding box defined: {}".format(self.bbox))

    def manually_label(self):
        if self.bbox is None or self.image_size is None:
            self.get_bbox()
        self.manual_labels = dict()
        dir_entries = list(os.scandir(self.images_dir))
        assert(len(dir_entries) >= self.num_train)
        assert(all([x.is_file() for x in dir_entries]))

        for i in r.sample(dir_entries, self.num_train):
            if SCALE_ALL_IMAGES:
                label = LI.manually_label(i.path, self.bbox, self.image_size)
            else:
                label = LI.manually_label(i.path, self.bbox)
            self.manual_labels[i.name] = label

    def crossvalidate(self):
        """
        leave-one-out crossvalidation
        """
        results = []
        full_label_set = set(self.manual_labels.values())
        d = sorted(self.manual_labels.keys())
        for i in range(len(d)):
            tmp_train = list(d)
            del tmp_train[i]
            tmp_label_set = set([self.manual_labels[x] for x in tmp_train])
            # no training samples for one of the classes, skip this
            if not len(tmp_label_set) == len(full_label_set):
                continue
            model = self.train_model({k: self.manual_labels[k] for k in tmp_train})
            cls = self.classify_image(model, d[i])
            print("cls: {}, prediction: {}".format(self.manual_labels[d[i]], cls))
            results.append(cls == self.manual_labels[d[i]])

        print(results)
        print("{} / {} correct".format(len([res for res in results if res]), len(results)))

    def get_feature_vector(self, image_filename):
        """
        uses self's images_dir and bounding box to extract image features
        """
        im = Image.open(self.images_dir+"/"+image_filename)
        if SCALE_ALL_IMAGES:
            im = im.resize(self.image_size)
        im = im.crop(self.bbox)
        (width, height) = im.size
        im = (1.0/256) * np.array(list(im.getdata())).reshape((height, width, 3))
        return feature.hog(color.rgb2gray(im)) # also, try daisy
 
    def train_model(self, labelled_training_dataset):
        """
        labelled_training_dataset is a dict mapping from filename -> label
        returns model
        """
        d = list(labelled_training_dataset.keys()) 
        X = [self.get_feature_vector(k) for k in d]
        y = [labelled_training_dataset[k] for k in d]
        classifier = svm.SVC(probability=True)
        classifier.fit(X, y)
        return classifier

    def evaluate_model(self, model, labelled_test_set):
        raise Exception("Need to re-implement this function, I don't trust scikit learn's predict method")
        d = list(labelled_test_dataset.keys()) 
        X = [self.get_feature_vector(k) for k in d]
        y_pred = model.predict(X)
        y_true = [labelled_test_dataset[k] for k in d]
        num_correct = len([i for i in range(len(labelled_test_dataset)) if y_pred[i] == y_true[i]])
        num_incorrect = len([i for i in range(len(labelled_test_dataset)) if not y_pred[i] == y_true[i]])
        return num_correct, num_incorrect

    def classify_image(self, model, image_filename):
        f = self.get_feature_vector(image_filename)
        X = [f]
        probs = model.predict_proba(X)[0]
        classes = sorted(model.classes_)
        return classes[max(range(len(classes)), key = lambda x: probs[x])]
        
    def __str__(self):
        return "ClassificationProject:\nbbox:{}\nnumber of manually labeled items:{}\nnumber of classes:{}".format(self.bbox, len(self.manual_labels), len(set(self.manual_labels.values())))

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
        project = ClassificationProject.from_file(args.output_dir+"/project.pkl")
        if project is None:
            project = ClassificationProject(args.output_dir+"/project.pkl", args.images_dir, NUM_TRAIN)
        if not len(project.manual_labels) == NUM_TRAIN:
            project.manually_label()
            project.save()
        print(project)
        project.crossvalidate()

    elif args.command == "label-blank":
        pass
    elif args.command == "summary":
        project = ClassificationProject.from_file(args.output_dir+"/project.pkl")
        print(project)
        
    else:
        print ("no, just no")

