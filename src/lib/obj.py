# amfs_tm/src/lib/obj.py
import os
import logging

class Feature(object):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        self.root = root
        self.data_path = data_path
        self.out_path = out_path
        self.sep = sep
        self.snapshot = snapshot
        
        # Absolute Volume Paths
        self.abs_data_path = os.path.join(self.root, self.data_path)
        self.abs_out_path = os.path.join(self.root, self.out_path)

    def safe_makedirs(self, path):
        """Safely creates directories in Databricks Volumes without Errno 95."""
        if not path or os.path.exists(path):
            return
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # If we hit the Volume root restriction, try creating the leaf directory
            if e.errno == 95: 
                parent = os.path.dirname(path)
                if os.path.exists(parent):
                    os.mkdir(path)
            else:
                raise e

    def create(self):
        pass