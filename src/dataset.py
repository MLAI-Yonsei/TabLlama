import torch
from torch.utils.data import Dataset
import random
from typing import List, Dict, Any, Tuple, Optional

class VehicleDataset(Dataset):
    def __init__(self, processed_data: List[Dict[str, Any]], transform: Optional[List[Any]] = None):
        self.data = processed_data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        data = item["data"]
        padding_mask = item["padding_mask"].bool()

        if self.transform:
            for tr in self.transform:
                data, padding_mask = tr(data, padding_mask)

        return {
            "data": data,
            "intent_label": item["intent_label"],
            "padding_mask": padding_mask,
            "intent": item["intent"],
        }

class VehicleInferDataset(Dataset):
    def __init__(self, args: Any, processed_data: List[Dict[str, Any]], transform: Optional[List[Any]] = None):
        self.transform = transform
        self.filtered_data = [item for item in processed_data if item["padding_mask"].sum() == 0]
        self.filtered_indices = [idx for idx, item in enumerate(processed_data) if item["padding_mask"].sum() == 0]

    def __len__(self) -> int:
        return len(self.filtered_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.filtered_data[idx]
        data = item["data"]
        padding_mask = item["padding_mask"].bool()

        if self.transform:
            for tr in self.transform:
                data, padding_mask = tr(data, padding_mask)

        return {
            "data": data,
            "intent_label": item["intent_label"],
            "padding_mask": padding_mask,
            "intent": item["intent"],
        }

class AddGaussianNoise:
    def __init__(self, max_value: float, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
        self.max_value = max_value

    def __call__(self, tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = torch.round(tensor + noise)
        return torch.clamp(noisy_tensor, min=0, max=self.max_value).to(torch.int), mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class AugmentWarp:
    def __init__(self, limit: int = -1):
        self.limit = limit
        
    def __call__(self, sample: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_list = list(range(len(sample)))
        false_indices = torch.where(~mask)[0].tolist()
        k = min(random.randint(0, int(mask.sum().item())), self.limit) if self.limit != -1 else random.randint(0, int(mask.sum().item()))
        
        sampled_indices = random.choices(false_indices, k=k)
        new_seq_list = sorted(seq_list + sampled_indices)

        new_stack = [sample[i] for i in new_seq_list]
        new_tensor = torch.stack(new_stack)
        new_mask = torch.cat([mask[k:], torch.tensor([False] * k)])

        return new_tensor[k:], new_mask.bool()

class SwapAugment:
    def __init__(self, swap_prob: float = 0.1):
        self.swap_prob = swap_prob

    def __call__(self, sample: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_list = list(range(len(sample)))

        for i in range(len(seq_list) - 1):
            if random.random() < self.swap_prob:
                seq_list[i], seq_list[i + 1] = seq_list[i + 1], seq_list[i]

        new_stack = [sample[i] for i in seq_list]
        new_tensor = torch.stack(new_stack)
        new_mask = mask.clone()

        return new_tensor, new_mask
