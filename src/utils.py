import random
import numpy as np
import torch
import time
import subprocess
from typing import List, Dict, Any, Tuple
from sklearn.metrics import precision_recall_fscore_support

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_memory_usage(device: torch.device) -> int:
    """Get GPU memory usage for the specified device."""
    if isinstance(device, str):
        if device.startswith('cuda:'):
            gpu_id = int(device.split(':')[1])
        elif device == 'cuda':
            gpu_id = 0
        else:
            return 0  # CPU or unknown device
    elif isinstance(device, torch.device):
        if device.type == 'cuda':
            gpu_id = device.index if device.index is not None else 0
        else:
            return 0  # CPU
    else:
        raise ValueError("Unsupported device type")

    result = subprocess.check_output(
        [
            'nvidia-smi', f'--id={gpu_id}', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    return int(result.strip())

def find_min_max_values_from_loader(data_loader) -> Tuple[float, float]:
    """Find minimum and maximum values in a data loader."""
    min_value = float('inf')
    max_value = float('-inf')
    
    for batch in data_loader:
        if 'data' in batch and isinstance(batch['data'], torch.Tensor):
            tensor_min = batch['data'].min().item()
            tensor_max = batch['data'].max().item()
            min_value = min(min_value, tensor_min)
            max_value = max(max_value, tensor_max)
    
    return min_value, max_value

def find_max_value(data_list: List[Dict[str, Any]]) -> float:
    """Find maximum value in a list of data dictionaries."""
    return max(item['data'].max().item() for item in data_list if 'data' in item and isinstance(item['data'], torch.Tensor))

def run_inference_bert(
    model: torch.nn.Module,
    model_path: str,
    inference_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sequence_len: int,
    initial_sequence_len: int = 5,
    num_classes: int = None,
) -> Tuple[float, float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]], float, int]:
    """
    Run inference on the given model and calculate various metrics.

    Args:
        model: The model to run inference on.
        model_path: Path to the saved model weights.
        inference_dataloader: DataLoader for inference data.
        device: Device to run inference on.
        sequence_len: Total sequence length.
        initial_sequence_len: Initial sequence length for prediction.
        num_classes: Number of intent classes.

    Returns:
        Tuple containing various metrics and predictions.
    """
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_true_values = []
    all_intent_predictions = []
    all_true_intents = []
    sequence_wise_errors = [[] for _ in range(sequence_len - initial_sequence_len)]
    sequence_wise_intent_predictions = [[] for _ in range(sequence_len - initial_sequence_len)]
    sequence_wise_true_intents = [[] for _ in range(sequence_len - initial_sequence_len)]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for batch in inference_dataloader:
            inputs, targets, masks = [b.to(device) for b in (batch["data"], batch["intent_label"], batch["padding_mask"])]
            batch_size, _, feature_dim = inputs.shape
                        
            
            current_sequence = inputs[:, :initial_sequence_len, :]
            current_mask = masks[:, :initial_sequence_len]
            
            
            predictions = []

            for i in range(initial_sequence_len, sequence_len):
                output_seq, intent_output = model(current_sequence.long(), current_mask)
                
                last_prediction = torch.clamp(output_seq.unsqueeze(1), max=444)
                last_prediction_int = torch.round(last_prediction).long()
                predictions.append(last_prediction_int)

                true_value = inputs[:, i, :]
                error = (last_prediction_int.float().squeeze() - true_value.float()).abs().mean(dim=0).cpu().numpy()
                sequence_wise_errors[i - initial_sequence_len].append(error)

                _, intent_pred = intent_output.max(1)
                sequence_wise_intent_predictions[i - initial_sequence_len].extend(intent_pred.cpu().numpy())
                sequence_wise_true_intents[i - initial_sequence_len].extend(targets.cpu().numpy())

                current_sequence = torch.cat([current_sequence[:, :, :], last_prediction.float()], dim=1)
                current_mask = torch.cat([current_mask[:, :], torch.zeros((batch_size, 1), device=device)], dim=1)

            all_predictions.extend(torch.cat(predictions, dim=1).cpu().numpy())
            all_true_values.extend(inputs[:, initial_sequence_len:, :].cpu().numpy())

            _, final_intent_pred = intent_output.max(1)
            all_intent_predictions.extend(final_intent_pred.cpu().numpy())
            all_true_intents.extend(targets.cpu().numpy())

    torch.cuda.synchronize()
    end_time = time.time()
    execution_time = end_time - start_time
    max_memory_usage = torch.cuda.max_memory_allocated()

    all_predictions = np.array(all_predictions)
    all_true_values = np.array(all_true_values)
    all_intent_predictions = np.array(all_intent_predictions)
    all_true_intents = np.array(all_true_intents)

    # Calculate overall metrics
    mae = np.mean(np.abs(all_predictions - all_true_values))
    rmse = np.sqrt(np.mean((all_predictions - all_true_values) ** 2))
    seq_accuracy = np.mean(all_predictions == np.round(all_true_values))
    non_zero_mask = all_true_values != 0
    non_zero_acc = np.mean(all_predictions[non_zero_mask] == np.round(all_true_values[non_zero_mask]))
    non_zero_mae = np.mean(np.abs(all_predictions[non_zero_mask] - all_true_values[non_zero_mask]))
    non_zero_rmse = np.sqrt(np.mean((all_predictions[non_zero_mask] - all_true_values[non_zero_mask]) ** 2))
    intent_accuracy = np.mean(all_intent_predictions == all_true_intents)

    print(f"Overall Inference Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
          f"Non-zero MAE: {non_zero_mae:.4f}, Non-zero RMSE: {non_zero_rmse:.4f}, "
          f"Intent Accuracy: {intent_accuracy:.4f}")

    # Calculate sequence-wise metrics
    sequence_wise_metrics = []
    for i in range(len(sequence_wise_errors)):
        position = i + initial_sequence_len
        position_errors = np.array(sequence_wise_errors[i])
        position_mae = np.mean(position_errors)
        position_rmse = np.sqrt(np.mean(position_errors**2))
        position_seq_accuracy = np.mean(all_predictions[:, i, :] == np.round(all_true_values[:, i, :]))
        position_non_zero_mask = all_true_values[:, i, :] != 0
        position_non_zero_acc = np.mean(all_predictions[:, i, :][position_non_zero_mask] == np.round(all_true_values[:, i, :][position_non_zero_mask]))
        position_non_zero_mae = np.mean(np.abs(all_predictions[:, i, :][position_non_zero_mask] - all_true_values[:, i, :][position_non_zero_mask]))
        position_non_zero_rmse = np.sqrt(np.mean((all_predictions[:, i, :][position_non_zero_mask] - all_true_values[:, i, :][position_non_zero_mask]) ** 2))
        position_intent_accuracy = np.mean(np.array(sequence_wise_intent_predictions[i]) == np.array(sequence_wise_true_intents[i]))

        sequence_wise_metrics.append({
            "position": position,
            "mae": position_mae,
            "rmse": position_rmse,
            "non_zero_mae": position_non_zero_mae,
            "non_zero_rmse": position_non_zero_rmse,
            "intent_accuracy": position_intent_accuracy,
        })

        print(f"Position {position} - MAE: {position_mae:.4f}, RMSE: {position_rmse:.4f}, "
              f"Non-zero MAE: {position_non_zero_mae:.4f}, Non-zero RMSE: {position_non_zero_rmse:.4f}, "
              f"Intent Accuracy: {position_intent_accuracy:.4f}")

    if num_classes is not None:
        print("\nOverall Intent Classification Metrics:")
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_intents, all_intent_predictions, average=None, labels=range(num_classes))
        for i in range(num_classes):
            print(f"Class {i}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-score: {f1[i]:.4f}")

        print("\nSequence-wise Intent Classification Metrics:")
        for i, (intent_preds, true_intents) in enumerate(zip(sequence_wise_intent_predictions, sequence_wise_true_intents)):
            position = i + initial_sequence_len
            precision, recall, f1, _ = precision_recall_fscore_support(true_intents, intent_preds, average=None, labels=range(num_classes))
            print(f"\nPosition {position}:")
            for j in range(num_classes):
                print(f"Class {j}: Precision: {precision[j]:.4f}, Recall: {recall[j]:.4f}, F1-score: {f1[j]:.4f}")

    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Max GPU memory usage: {max_memory_usage / 1024 / 1024:.2f} MB")

    return (mae, rmse, non_zero_mae, non_zero_rmse, seq_accuracy, non_zero_acc, intent_accuracy,
            all_predictions, all_true_values, all_intent_predictions, all_true_intents,
            sequence_wise_metrics, execution_time, max_memory_usage)