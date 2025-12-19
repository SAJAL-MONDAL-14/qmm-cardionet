import os
import ast
import torch
import wfdb
import pandas as pd
from torch.utils.data import Dataset

class PTBXLDataset(Dataset):
    def __init__(self, csv_path, max_len=5000):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len

        # keep only rows with high-resolution ECG
        self.df = self.df[self.df["filename_hr"].notna()].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _label(self, scp_codes):
        codes = ast.literal_eval(scp_codes)
        return 0 if list(codes.keys()) == ["NORM"] else 1

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # IMPORTANT FIX HERE
        record_path = os.path.join("data/ptbxl", row["filename_hr"])

        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal[:, 0], dtype=torch.float32)

        if len(signal) > self.max_len:
            signal = signal[:self.max_len]
        else:
            signal = torch.nn.functional.pad(
                signal, (0, self.max_len - len(signal))
            )

        label = torch.tensor(
            self._label(row["scp_codes"]),
            dtype=torch.float32
        )

        return signal, label
