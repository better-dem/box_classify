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

possible_commands = {"full": "run all steps, from manual labeling through classifying all remaining examples, ignoring steps that are already saved in the project file", 
                     "manual-label": "run the manual labeling step", "train":"run the training step, requires manual labeling to be done first", 
                     "x-validate": "run leave-one-out cross-validation on manually-labeled samples", 
                     "summary":"Load a project from the output directory and summarize it", 
                     "write-csv":"Write out a csv of label results. Can include multiple project files and there will only be one row per item"}


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Run box_classifier', epilog = "Possible commands: \n"+"\n".join([" - "+i[0] + ": " + i[1] for i in possible_commands.items()]))

parser.add_argument('-c', '--command', required=True, type=str, help="the function to run. One of these:" + str(possible_commands.keys()), action='store')
parser.add_argument('-i', '--images_dir', required=False, type=str, help="the directory containing only image files to be used for this classification", action='store')
parser.add_argument('-p', '--project_files', nargs='+', required=True, type=str, help="the file to store pickled project object", action='store')
parser.add_argument('-o', '--output_file', required=False, type=str, help="the file to save output such as csv files", action='store')

args = parser.parse_args()


def write_results_csv(named_classfication_projects, filename):
    """
    write out results from several classification projects to a single CSV file
    join by item names

    Params:
    named_classfication_projects: a dict, mapping project name to project object
    filename: csv output filename
    """
    with open(filename,'w') as f:
        title_row_items = ["item name"] 
        for n in sorted(named_classfication_projects.keys()):
            title_row_items += [n + " label", n + " label origin"]
        f.write(','.join(title_row_items)+"\n")
        
        # order item names with manually labelled items at the top
        all_item_names = []
        for p in named_classfication_projects.values():
            all_item_names = [k for k in p.manual_labels.keys() if not k in all_item_names] + all_item_names
            all_item_names = all_item_names + [k for k in p.result_labels.keys() if not k in all_item_names]

        for item in all_item_names:
            row = [item]
            for project_name in sorted(named_classfication_projects.keys()):
                project = named_classfication_projects[project_name]
                if item in project.manual_labels:
                    row += [ project.manual_labels[item], "manually labelled"]
                elif item in project.result_labels:
                    row += [ project.result_labels[item], "automatically labelled"]
                else:
                    row += ["","NA"]
            f.write(','.join(row)+"\n")


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
            results.append(cls == self.manual_labels[d[i]])

        print(results)
        print("{} / {} correct".format(len([res for res in results if res]), len(results)))

    def train(self):
        """
        train the full model with all manually labelled date
        """
        self.model = self.train_model(self.manual_labels)

    def classify_all(self, upto=None):
        dir_entries = list(os.scandir(self.images_dir))
        assert(len(dir_entries) >= self.num_train)
        assert(all([x.is_file() for x in dir_entries]))
        work_set = [e.name for e in dir_entries if not e.name in self.manual_labels.keys()]
        print("Classifying all unlabelled images. Number of images: {}".format(len(work_set)))

        j = 0
        for item in work_set:
            if not upto is None and j >= upto:
                break
            if not j == 0 and j % 100 == 0:
                print("On image # {}".format(j))
            self.result_labels[item] = self.classify_image(self.model, item)
            j += 1

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
        return "ClassificationProject:\nbbox:{}\nnumber of manually labeled items:{}\nnumber of automatically labeled items:{}\nnumber of classes:{}".format(self.bbox, len(self.manual_labels), len(self.result_labels), len(set(self.manual_labels.values())))

if __name__ == "__main__":
    if args.command == "full":
        assert(len(args.project_files) == 1)
        project = ClassificationProject.from_file(args.project_files[0])
        if project is None:
            project = ClassificationProject(args.project_files[0], args.images_dir, NUM_TRAIN)

        if not len(project.manual_labels) == NUM_TRAIN:
            project.manually_label()
            project.save()

        if project.model is None:
            project.train()
            project.save()

        print("classifying all non-manually-labelled images ...")
        project.classify_all()
        project.save()
        print("done")

    elif args.command == "manual-label":
        assert(len(args.project_files) == 1)
        project = ClassificationProject(args.project_files[0], args.images_dir, NUM_TRAIN)
        project.manually_label()
        project.save()
        print(project)
        
    elif args.command == "x-validate":
        assert(len(args.project_files) == 1)
        project = ClassificationProject.from_file(args.project_files[0])
        if project is None:
            project = ClassificationProject(args.project_files[0], args.images_dir, NUM_TRAIN)
        if not len(project.manual_labels) == NUM_TRAIN:
            project.manually_label()
            project.save()
        print(project)
        project.crossvalidate()

    elif args.command == "summary":
        for p in args.project_files:
            project = ClassificationProject.from_file(p)
            print(project)

    elif args.command == "write-csv":
        print(args.project_files)
        named_projects = {x.split('/')[-1]: ClassificationProject.from_file(x) for x in args.project_files}
        write_results_csv(named_projects, args.output_file)

    else:
        print ("no, just no")

