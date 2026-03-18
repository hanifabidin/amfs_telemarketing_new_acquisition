# amfs_tm/src/lib/feature/geo/cust_zipcode.py
import os
import numpy as np
import pandas as pd
from lib import obj, util

class CustZipcode(obj.Feature):
    def __init__(self, root, data_path, out_path, sep, snapshot):
        super().__init__(root, data_path, out_path, sep, snapshot)

        # Paths updated for Databricks Volume structure
        self.zip_path = os.path.join(self.root, data_path, snapshot, 'axa_zipcode_{0}.csv')
        self.feature_path = os.path.join(self.root, out_path, 'zipcode_{0}_feat.csv')

    def create(self):
        input_path = self.zip_path.format(self.snapshot)
        print(f'Reading Zipcode data: {input_path}')

        if not os.path.exists(input_path):
            return

        # Load data - usually contains CIF and ZIP code
        df = pd.read_csv(input_path, sep=self.sep)
        df.columns = [c.lower() for c in df.columns]

        # Standardizing primary key
        if 'cifno' not in df.columns and 'cus_no' in df.columns:
            df = df.rename(columns={'cus_no': 'cifno'})

        util.to_numeric(df, np.int64, 'cifno')

        # Cleaning Zipcodes (ensuring they are strings of 5 digits)
        if 'zipcode' in df.columns:
            df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
            
            # Example Geography Mapping logic often found in your repo:
            # First 2 digits of ZIP often represent the region/province in Indonesia
            df['zip_region'] = df['zipcode'].str[:2]
            
            # Identify Jabodetabek region (starts with 10-17)
            df['is_jabodetabek'] = df['zip_region'].isin([str(i) for i in range(10, 18)]).astype(int)

        output_file = self.feature_path.format(self.snapshot)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f'Saving Zipcode features to {output_file}')
        df.to_csv(output_file, index=False)