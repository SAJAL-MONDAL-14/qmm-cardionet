import os
import ast
import torch
import wfdb
import pandas as pd
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, clinical_csv, ptbxl_csv, max_len=5000):
        self.clinical = pd.read_csv(clinical_csv)
        self.ecg = pd.read_csv(ptbxl_csv)

        # Keep only high-resolution ECG
        self.ecg = self.ecg[self.ecg["filename_hr"].notna()]

        # Align length (simple baseline)
        min_len = min(len(self.clinical), len(self.ecg))
        self.clinical = self.clinical.iloc[:min_len]
        self.ecg = self.ecg.iloc[:min_len]

        self.max_len = max_len

    def __len__(self):
        return len(self.clinical)

    def _label(self, scp_codes):
        codes = ast.literal_eval(scp_codes)
        return 0 if list(codes.keys()) == ["NORM"] else 1

    def __getitem__(self, idx):
        # -------- Clinical --------
        clinical_x = torch.tensor(
            self.clinical.iloc[idx].values[:-1],
            dtype=torch.float32
        )

        # -------- ECG --------
        record_path = os.path.join(
            "data/ptbxl",
            self.ecg.iloc[idx]["filename_hr"]
        )

        signal, _ = wfdb.rdsamp(record_path)
        signal = torch.tensor(signal[:, 0], dtype=torch.float32)

        if len(signal) > self.max_len:
            signal = signal[:self.max_len]
        else:
            signal = torch.nn.functional.pad(
                signal, (0, self.max_len - len(signal))
            )

        label = torch.tensor(
            self._label(self.ecg.iloc[idx]["scp_codes"]),
            dtype=torch.float32
        )

        return clinical_x, signal, label
