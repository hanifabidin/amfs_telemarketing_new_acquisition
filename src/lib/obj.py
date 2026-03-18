# amfs_tm/src/lib/obj.py
import abc
import os

class Feature(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, root, data_path, out_path, sep, snapshot):
        """
        In Databricks:
        root: Absolute path to the project (e.g., /Volumes/catalog/schema/volume/amfs_tm)
        data_path: Absolute path to processed data
        out_path: Absolute path for feature output
        """
        self.root = root
        self.data_path = data_path
        self.out_path = out_path
        self.sep = "|"
        self.snapshot = snapshot

    @abc.abstractmethod
    def create(self):
        pass