import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['Cost2100DataLoader', 'PreFetcher']


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input


class Cost2100DataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.

    This loader can concatenate multiple scenarios ("in","out","dm").
    Pass `scenario` as a single string, a comma-separated string, or a list.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, scenario):
        assert os.path.isdir(root)
        # normalize scenario argument to list of strings
        if isinstance(scenario, str) and "," in scenario:
            scenario = [s.strip() for s in scenario.split(",") if s.strip()]
        if isinstance(scenario, str):
            scenarios = [scenario]
        elif isinstance(scenario, (list, tuple)):
            scenarios = list(scenario)
        else:
            raise ValueError("scenario must be a string or list of strings")
        for sc in scenarios:
            assert sc in {"in", "out", "dm"}, f"unknown scenario {sc}"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        channel, nt, nc, nc_expand = 2, 32, 32, 125

        # accumulate tensors for each split
        train_list = []
        val_list = []
        test_list = []
        raw_list = []

        for sc in scenarios:
            dir_train = os.path.join(root, f"DATA_Htrain{sc}.mat")
            dir_val = os.path.join(root, f"DATA_Hval{sc}.mat")
            dir_test = os.path.join(root, f"DATA_Htest{sc}.mat")
            dir_raw = os.path.join(root, f"DATA_HtestF{sc}_all.mat")

            # Training data
            data_train = sio.loadmat(dir_train)["HT"]
            data_train = torch.tensor(data_train, dtype=torch.float32).view(
                data_train.shape[0], channel, nt, nc)
            train_list.append(data_train)

            # Validation data
            data_val = sio.loadmat(dir_val)["HT"]
            data_val = torch.tensor(data_val, dtype=torch.float32).view(
                data_val.shape[0], channel, nt, nc)
            val_list.append(data_val)

            # Test data (sparse)
            data_test = sio.loadmat(dir_test)["HT"]
            data_test = torch.tensor(data_test, dtype=torch.float32).view(
                data_test.shape[0], channel, nt, nc)
            test_list.append(data_test)

            # Raw test signal
            raw_test = sio.loadmat(dir_raw)["HF_all"]
            real = torch.tensor(np.real(raw_test), dtype=torch.float32)
            imag = torch.tensor(np.imag(raw_test), dtype=torch.float32)
            raw_test = torch.cat((
                real.view(raw_test.shape[0], nt, nc_expand, 1),
                imag.view(raw_test.shape[0], nt, nc_expand, 1)),
                dim=3)
            raw_list.append(raw_test)

        # concatenate along the sample dimension if multiple scenarios
        if len(train_list) > 1:
            data_train = torch.cat(train_list, dim=0)
            data_val = torch.cat(val_list, dim=0)
            data_test = torch.cat(test_list, dim=0)
            raw_test = torch.cat(raw_list, dim=0)
        else:
            data_train = train_list[0]
            data_val = val_list[0]
            data_test = test_list[0]
            raw_test = raw_list[0]

        self.train_dataset = TensorDataset(data_train)
        self.val_dataset = TensorDataset(data_val)
        self.test_dataset = TensorDataset(data_test, raw_test)

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            val_loader = PreFetcher(val_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, val_loader, test_loader
    
