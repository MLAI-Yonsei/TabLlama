import numpy as np
import pandas as pd
import torch
from lxml import etree
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import random
import math

class DataPreprocessor:
    @staticmethod
    def load_xml_to_dataframes(file_path: str) -> Dict[str, pd.DataFrame]:
        tree = etree.parse(file_path)
        root = tree.getroot()
        dataframes = {}

        for df_element in root.findall("dataframe"):
            df_name = df_element.get("name")
            data = []

            for row_element in df_element.findall("row"):
                row_data = {}
                for child in row_element:
                    column_name = child.tag[4:] if child.tag.startswith("col_") else child.tag
                    column_name = int(column_name) if column_name.isdigit() else column_name
                    row_data[column_name] = child.text
                data.append(row_data)

            df = pd.DataFrame(data)
            if "index" in df.columns:
                df.set_index("index", inplace=True)

            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass

            dataframes[df_name] = df

        return dataframes

    @staticmethod
    def create_intent_mapping(dataframes: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, int], int]:
        all_intents = set(f"{key.split('_')[1]}_{key.split('_')[2]}" for key in dataframes.keys())
        intent_mapping = {intent: idx for idx, intent in enumerate(sorted(all_intents))}
        return intent_mapping, len(all_intents)

    @staticmethod
    def prepare_data(dataframes: Dict[str, pd.DataFrame]) -> Tuple[List[Dict[str, Any]], float, int, int, Dict[str, int], int]:
        processed_data = []
        max_value = float("-inf")
        data_dim = max(df.shape[0] for df in dataframes.values())
        max_seq_len = max(df.shape[1] for df in dataframes.values())
        intent_mapping, num_intent = DataPreprocessor.create_intent_mapping(dataframes)

        print(f"Data dimension: {data_dim}, Max sequence length: {max_seq_len}, num_intent: {num_intent}")

        for key, df in dataframes.items():
            intent = f"{key.split('_')[1]}_{key.split('_')[2]}"
            intent_label = intent_mapping[intent]

            data = df.values.T.astype(float)
            seq_len, _ = data.shape

            padded_data = np.zeros((max_seq_len, data_dim))
            padded_data[-seq_len:, :] = data
            padded_data = np.clip(padded_data, a_min=None, a_max=444)

            padding_mask = np.ones(max_seq_len)
            padding_mask[-seq_len:] = 0

            max_value = max(max_value, np.max(padded_data))

            processed_data.append({
                "intent": intent,
                "intent_label": intent_label,
                "data": torch.FloatTensor(padded_data),
                "padding_mask": torch.FloatTensor(padding_mask),
            })

        return processed_data, max_value, data_dim, max_seq_len, intent_mapping, num_intent

    @staticmethod
    def split_data(processed_data: List[Dict[str, Any]], train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        data = [item["data"] for item in processed_data]
        labels = [item["intent_label"] for item in processed_data]

        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data, labels, train_size=train_size, stratify=labels, random_state=random_state
        )

        val_size_adjusted = val_size / (val_size + test_size)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, train_size=val_size_adjusted, stratify=temp_labels, random_state=random_state
        )

        def reconstruct_data(data, labels):
            return [
                {
                    "data": d,
                    "intent_label": l,
                    "padding_mask": processed_data[i]["padding_mask"],
                    "intent": processed_data[i]["intent"],
                }
                for i, (d, l) in enumerate(zip(data, labels))
            ]

        return reconstruct_data(train_data, train_labels), reconstruct_data(val_data, val_labels), reconstruct_data(test_data, test_labels)

    @staticmethod
    def indist_split_data(processed_data: List[Dict[str, Any]], val_size: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        def generate_random_sizes() -> Tuple[float, float, float]:
            remaining = 1 - val_size
            train_size = np.random.uniform(0.4, min(0.4, remaining))
            test_size = remaining - train_size
            if test_size < 0.2:
                overflow = 0.2 - test_size
                test_size = 0.2
                train_size -= overflow
            return train_size, val_size, test_size

        def split_sequence(seq_length: int, ratios: List[float]) -> List[int]:
            split_points = np.cumsum(np.array(ratios) * seq_length).astype(int)
            return [0] + list(split_points)

        def pad_left(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
            pad_width = target_length - tensor.shape[0]
            return torch.nn.functional.pad(tensor, (0, 0, pad_width, 0), mode="constant", value=0)

        train_data, val_data, test_data = [], [], []

        for item in processed_data:
            data = item["data"]
            padding_mask = item["padding_mask"]

            S = data.shape[0]

            seq_length = S - int(torch.sum(padding_mask))
            train_size, val_size, test_size = generate_random_sizes()
            split_indices = split_sequence(seq_length, [train_size, val_size, test_size])

            train_seq = data[split_indices[0]:split_indices[1]]
            val_seq = data[split_indices[1]:split_indices[2]]
            test_seq = data[split_indices[2]:split_indices[3]]

            train_mask = torch.zeros(S)
            train_mask[:S - train_seq.shape[0]] = 1
            val_mask = torch.zeros(S)
            val_mask[:S - val_seq.shape[0]] = 1
            test_mask = torch.zeros(S)
            test_mask[:S - test_seq.shape[0]] = 1

            assert val_mask.sum() < 30 and test_mask.sum() < 30, "Invalid mask sum"

            train_data.append(DataPreprocessor._create_data_item(item, pad_left(train_seq, S), train_mask))
            val_data.append(DataPreprocessor._create_data_item(item, pad_left(val_seq, S), val_mask))
            test_data.append(DataPreprocessor._create_data_item(item, pad_left(test_seq, S), test_mask))

        return train_data, val_data, test_data

    @staticmethod
    def _create_data_item(item: Dict[str, Any], data: torch.Tensor, mask: torch.Tensor) -> Dict[str, Any]:
        return {
            "intent": item["intent"],
            "intent_label": item["intent_label"],
            "data": data,
            "padding_mask": mask,
        }

    @staticmethod
    def augment_sequence_data(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        augmented_data = []

        for item in data_list:
            original_data = item["data"]
            original_mask = item["padding_mask"]

            augmented_data.append(item.copy())

            zero_count = torch.sum(original_mask == 0).item()

            for i in range(1, zero_count - 1):
                new_item = item.copy()
                new_item["data"] = torch.roll(original_data, shifts=i, dims=0)
                new_item["data"][:i] = 0
                new_item["padding_mask"] = torch.roll(original_mask, shifts=i)
                new_item["padding_mask"][:i] = 1
                augmented_data.append(new_item)

        return augmented_data

    @staticmethod
    def augment_sequence_data_indist(data_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        def split_list_random(input_list: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
            k = len(input_list)
            min_elements = math.ceil(k * 0.2)
            middle_start = random.randint(min_elements, k - 2*min_elements)
            middle_end = middle_start + min_elements
            return input_list[:middle_start], input_list[middle_start:middle_end], input_list[middle_end:]

        augmented_train_data, augmented_val_data, augmented_test_data = [], [], []

        for item in data_list:
            augmented_data = [item.copy()]
            original_data = item["data"]
            original_mask = item["padding_mask"]
            zero_count = torch.sum(original_mask == 0).item()

            for i in range(1, zero_count - 1):
                new_item = item.copy()
                new_item["data"] = torch.roll(original_data, shifts=i, dims=0)
                new_item["data"][:i] = 0
                new_item["padding_mask"] = torch.roll(original_mask, shifts=i)
                new_item["padding_mask"][:i] = 1
                augmented_data.append(new_item)

            te, va, tr = split_list_random(augmented_data)
            augmented_train_data.extend(tr)
            augmented_val_data.extend(va)
            augmented_test_data.extend(te)

        return augmented_train_data, augmented_val_data, augmented_test_data