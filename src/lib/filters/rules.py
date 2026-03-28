import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from lib import util

class FilterRule:
    def __init__(self, root, config, args):
        """
        Final Business Rules Engine for Leads Generation.
        root: base_path from main_config.yaml
        config: dictionary from filters_config.yaml
        args: Namespace with snapshot, dil, product
        """
        self.root = root
        self.cfg = config['rules']
        self.snapshot = args.snapshot
        self.dil = args.dil
        self.product = args.product
        
        # Date Calculations
        self.snapshot_dt = datetime.strptime(self.snapshot, '%Y%m')
        # Campaign is Snapshot + 1 Month (e.g., 202505 -> 202506)
        self.campaign = (self.snapshot_dt + relativedelta(months=1)).strftime('%Y%m')
        # campaign2 is next month in MMYY format (used for specific raw filenames)
        self.campaign2 = util.next_month_my(self.snapshot)
        
        # Output Pattern: /04_campaigns/{campaign}/01_filters/filtered_cif_{snapshot}_DIL_{DIL}_campaign_{campaign}.csv
        out_pattern = config['output']['path']
        self.output_path = os.path.join(self.root, out_pattern.format(
            campaign=self.campaign, 
            snapshot=self.snapshot, 
            DIL=self.dil
        ))

    def _load_df(self, rule_name, custom_snapshot=None):
        """
        Hardened loader for Volume files. 
        Auto-detects sep, standardizes columns to lowercase, and keys to 'cifno'.
        """
        rule = self.cfg.get(rule_name)
        if not rule or not rule.get('enabled'):
            return None
        
        target_snap = custom_snapshot if custom_snapshot else self.snapshot
        path = os.path.join(self.root, rule['file'].format(
            snapshot=target_snap, campaign2=self.campaign2
        ))
        
        if not os.path.exists(path):
            return None

        # Logic for separators
        sep = rule.get('sep')
        if not sep:
            sep = '\t' if path.endswith('.txt') else ','
            
        cols = [c.strip() for c in rule['usecols'].split(',')]
        
        try:
            df = pd.read_csv(path, sep=sep, usecols=cols)
            df.columns = df.columns.str.lower().str.strip()
            
            # Key standardization
            key_col = rule['key'].lower()
            if key_col in df.columns and key_col != 'cifno':
                df = df.rename(columns={key_col: 'cifno'})
            
            if 'cifno' in df.columns:
                util.to_numeric(df, np.int64, 'cifno')
            
            return df
        except Exception as e:
            print(f"Error reading {rule_name}: {e}")
            return None

    def apply(self):
        """Applies all business rules in sequence."""
        print(f"--- Starting Rules Engine: {self.snapshot} ---")
        
        # 0. Base Population (From PHSUMM Features)
        base_df = self._load_df('exclude_new_customer')
        if base_df is None:
            print("CRITICAL: exclude_new_customer (PHSUMM) source not found.")
            return None
        final_cifs = base_df[['cifno']].drop_duplicates()

        # 1. New Customer (Vintage) Filter
        min_v = self.cfg['exclude_new_customer']['params']['min_vint_months']
        exclude_v = base_df[base_df['saving_acct_vint_l'] < min_v]['cifno']
        final_cifs = final_cifs[~final_cifs['cifno'].isin(exclude_v)]
        print(f"Step 1 (Vintage) complete: {len(final_cifs)} remaining.")

        # 2. Segment & AXA Flag Filter
        seg_df = self._load_df('segment_rule')
        if seg_df is not None:
            p = self.cfg['segment_rule']['params']
            if p.get('exclude_non_active'):
                # bank_active_flag=0 means non-active
                final_cifs = final_cifs[~final_cifs['cifno'].isin(seg_df[seg_df['bank_active_flag'] == 0]['cifno'])]
            if p.get('exclude_existing_axa'):
                # axa_pol_flag=1 means existing policyholder
                final_cifs = final_cifs[~final_cifs['cifno'].isin(seg_df[seg_df['axa_pol_flag'] == 1]['cifno'])]
        print(f"Step 2 (Segments) complete: {len(final_cifs)} remaining.")

        # 3. Master Filter (Multi-Flag Criteria)
        master_df = self._load_df('master_filter')
        if master_df is not None:
            p = self.cfg['master_filter']['params']
            mask = pd.Series([True] * len(master_df), index=master_df.index)
            # Dynamically check all flags defined in YAML
            for col, val in p.items():
                if col in master_df.columns:
                    mask &= (master_df[col] == val)
            final_cifs = final_cifs[final_cifs['cifno'].isin(master_df[mask]['cifno'])]
        print(f"Step 3 (Master Filter) complete: {len(final_cifs)} remaining.")

        # 4. Consent Filter
        consent_df = self._load_df('consent_filter')
        if consent_df is not None:
            val = self.cfg['consent_filter']['params']['consent_value']
            eligible = consent_df[consent_df['consent_komunikasi_voice_call'] == val]['cifno']
            final_cifs = final_cifs[final_cifs['cifno'].isin(eligible)]
        print(f"Step 4 (Consent) complete: {len(final_cifs)} remaining.")

        # 5. Balance Filter (AVGD)
        bal_df = self._load_df('min_balance_rule')
        if bal_df is not None:
            min_b = self.cfg['min_balance_rule']['params']['min_avgd']
            eligible = bal_df[bal_df['avgd'] >= min_b]['cifno']
            final_cifs = final_cifs[final_cifs['cifno'].isin(eligible)]
        print(f"Step 5 (Balance) complete: {len(final_cifs)} remaining.")

        # 6. Age Tier Filter
        age_df = self._load_df('age_tier_rule')
        if age_df is not None:
            allowed = self.cfg['age_tier_rule']['params']['allowed_tiers']
            eligible = age_df[age_df['age_tier'].isin(allowed)]['cifno']
            final_cifs = final_cifs[final_cifs['cifno'].isin(eligible)]
        print(f"Step 6 (Age Tier) complete: {len(final_cifs)} remaining.")

        # 7. Bad Calls History (Multi-month lookup)
        bad_rule = self.cfg['bad_calls_history']
        if bad_rule['enabled']:
            bad_ids = bad_rule['params']['bad_callids']
            all_bad = set()
            for i in range(bad_rule['params']['months_back']):
                h_snap = (self.snapshot_dt - relativedelta(months=i)).strftime('%Y%m')
                df_h = self._load_df('bad_calls_history', custom_snapshot=h_snap)
                if df_h is not None:
                    all_bad.update(df_h[df_h['callid'].isin(bad_ids)]['cifno'].tolist())
            final_cifs = final_cifs[~final_cifs['cifno'].isin(all_bad)]
        print(f"Step 7 (Bad Calls History) complete: {len(final_cifs)} remaining.")

        # 8. Do Not Call (DNC) List
        dnc_df = self._load_df('do_not_call')
        if dnc_df is not None:
            final_cifs = final_cifs[~final_cifs['cifno'].isin(dnc_df['cifno'])]
        print(f"Step 8 (DNC) complete: {len(final_cifs)} remaining.")

        # 9. Sent Leads History (Exclude recent interactions)
        sent_df = self._load_df('sent_leads_history')
        if sent_df is not None:
            m_back = self.cfg['sent_leads_history']['params']['exclude_months']
            cutoff = self.snapshot_dt - relativedelta(months=m_back)
            sent_df['sent_date'] = pd.to_datetime(sent_df['sent_date'])
            exclude = sent_df[sent_df['sent_date'] >= cutoff]['cifno']
            final_cifs = final_cifs[~final_cifs['cifno'].isin(exclude)]
        print(f"Step 9 (Sent Leads History) complete: {len(final_cifs)} remaining.")

        # 10. Call Tracking History (Exclude Contacted/Converted)
        ct_rule = self.cfg['call_tracking_history']
        if ct_rule['enabled']:
            all_ct = set()
            for i in range(ct_rule['params']['months_back']):
                h_snap = (self.snapshot_dt - relativedelta(months=i)).strftime('%Y%m')
                df_c = self._load_df('call_tracking_history', custom_snapshot=h_snap)
                if df_c is not None:
                    mode = ct_rule['params']['exclude_mode']
                    thresh = 901 if mode == 'converted' else 201 if mode == 'contacted' else 0
                    all_ct.update(df_c[df_c['callid'] >= thresh]['cifno'].tolist())
            final_cifs = final_cifs[~final_cifs['cifno'].isin(all_ct)]
        print(f"Step 10 (Call Tracking History) complete: {len(final_cifs)} remaining.")

        # Final Save and Directory Creation
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        final_cifs.drop_duplicates().to_csv(self.output_path, index=False)
        
        print(f"\n✅ Filter complete. Output File: {self.output_path}")
        return self.output_path