"""
    script to classify galaxy merger detection task using
    different classifiers
    python classification_model.py dataset/*/training/*jpeg
        dataset/*/validation/*jpeg
        dataset/*/test/*jpeg
        model_configs.json
"""

import argparse
import json
import logging

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utils import load_data
from utils import load_json


class ClassificationModel:
    """
        This class fits and evaluates training data using sklearn classifiers

    Parameters
    ----------
    model_configs
        json file with classification model configs, check example config file
    """

    def __init__(self,
                 model_configs: dict):
        self.model_configs = model_configs
        self._model_name_mappings = {
            "random_forest": RandomForestClassifier,
            "extra_trees": ExtraTreesClassifier,
            "multilayer_perceptron": MLPClassifier,
            "decision_tree": DecisionTreeClassifier,
            "logistic": LogisticRegression,
            "qda": QuadraticDiscriminantAnalysis,
            "knn": KNeighborsClassifier
        }
        self._scaler = StandardScaler()
        self._classifier = None
        self._evaluation_results = {}

    def fit_model(self, train_data, train_labels):
        self._scaler.fit(train_data)
        train_data = self._scaler.transform(train_data)
        base_model = self._model_name_mappings[self.model_configs["model_name"]]
        self._classifier = base_model(**self.model_configs["model_params"])
        self._classifier.fit(train_data, train_labels)

    def evaluate_model(self, evaluation_data, evaluation_labels, mode):
        evaluation_data = self._scaler.transform(evaluation_data)
        test_calibrated_probs = self._classifier.predict_proba(
            evaluation_data)
        test_predictions = np.argmax(test_calibrated_probs, axis=1)
        evaluation_report = classification_report(
            evaluation_labels, test_predictions,
            target_names=self.model_configs["class_names"], output_dict=True)
        self._evaluation_results[mode] = evaluation_report

    def save_results(self):
        save_name = self.model_configs["model_name"] + ".json"
        with open(save_name, 'w') as json_file:
            json.dump(self._evaluation_results, json_file, indent=4)


def main(train_images_path: str, val_images_path: str, test_images_path: str,
         model_configs_path: str):
    model_configs = load_json(model_configs_path)
    class_mappings = model_configs["class_mappings"]
    image_size = model_configs["image_size"]
    logging.info("Loading training data")
    train_data, train_labels = load_data(
        train_images_path, class_mappings, image_size)
    logging.info("Loading validation data")
    val_data, val_labels = load_data(
        val_images_path, class_mappings, image_size)
    logging.info("Loading test data")
    test_data, test_labels = load_data(
        test_images_path, class_mappings, image_size)
    classification_class = ClassificationModel(
        model_configs=model_configs
    )
    logging.info(f"Fitting the training data using"
                 f" {model_configs['model_name']} classifier")
    classification_class.fit_model(train_data, train_labels)
    logging.info("Evaluating the training data")
    classification_class.evaluate_model(train_data, train_labels, "train")
    logging.info("Evaluating the validation data")
    classification_class.evaluate_model(val_data, val_labels, "val")
    logging.info("Evaluating the test data")
    classification_class.evaluate_model(test_data, test_labels, "test")
    logging.info("Saving evaluation results")
    classification_class.save_results()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='classifier script')
    parser.add_argument("train_images_path", type=str,
                        help="The path to train images")
    parser.add_argument("val_images_path", type=str,
                        help="The path to validation images")
    parser.add_argument("test_images_path", type=str,
                        help="The path to test images")
    parser.add_argument("model_configs_path", type=str,
                        help="The path to model config json")
    args = parser.parse_args()
    main(args.train_images_path, args.val_images_path,
         args.test_images_path, args.model_configs_path)
