import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from utils import set_seed, run_inference_bert
from data_processor import DataPreprocessor
from dataset import VehicleDataset, AddGaussianNoise, AugmentWarp, SwapAugment, VehicleInferDataset
from trainer import Trainer
from models.tabbert import HierarchicalTransformer
from models.lstmrnn import SimplifiedLSTMRNN
from models.vanillatf import SimplifiedHierarchicalTransformer
from models.tabgpt2 import HierarchicalGPT2
from models.tabllama import HierarchicalLlama

# 환경 변수 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 설정 스키마 정의
CONFIG_SCHEMA = {
    "model_type": {"type": str, "default": "tabllama", "choices": ["lstm", "rnn", "vanillatf", "tabbert", "tabgpt2", "tabllama"]},
    "hidden_size": {"type": int, "default": 4},
    "num_layers": {"type": int, "default": 2},
    "seq_num_layers": {"type": int, "default": 2},
    "num_heads": {"type": int, "default": 4},
    "seq_num_heads": {"type": int, "default": 16},
    "intermediate_size": {"type": int, "default": 2400},
    "num_epochs": {"type": int, "default": 100},
    "learning_rate": {"type": float, "default": 0.0001},
    "file_path": {"type": str, "default": "path/to/your/data.xml"},
    "batch_size": {"type": int, "default": 32},
    "col_dim": {"type": int, "default": 5},
    "use_augment_warp": {"type": bool, "default": False},
    "use_random_noise": {"type": bool, "default": False},
    "use_augment_swap": {"type": bool, "default": False},
    "swap_prob": {"type": float, "default": 0.2},
    "noise_mean": {"type": float, "default": 0},
    "noise_std": {"type": float, "default": 0.1},
    "warp_limit": {"type": int, "default": 20},
    "seed": {"type": int, "default": 0},
    "patience": {"type": int, "default": 10},
    "min_delta": {"type": float, "default": 0.001},
    "debug": {"type": bool, "default": False},
    "train": {"type": bool, "default": True},
    "infer": {"type": bool, "default": False},
    "test": {"type": bool, "default": False},
    "in_dist": {"type": bool, "default": False},
    "test_swap": {"type": bool, "default": False},
    "init_seq": {"type": int, "default": 20},
    "load_path": {"type": str, "default": "best_model.pth"},
    "save_path": {"type": str, "default": "best_model.pth"},
    "device": {"type": str, "default": "cuda"},
}

def parse_args():
    """설정 파일과 명령줄 인자를 파싱하여 실행 설정을 생성합니다."""
    parser = argparse.ArgumentParser(description="Vehicle Data Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    
    for key, value in CONFIG_SCHEMA.items():
        kwargs = {"type": value["type"], "default": value["default"], "help": f"{key} (default: {value['default']})"}
        if "choices" in value:
            kwargs["choices"] = value["choices"]
        parser.add_argument(f"--{key}", **kwargs)

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if key in CONFIG_SCHEMA:
                setattr(args, key, value)
            else:
                print(f"Warning: Unknown configuration key in YAML: {key}")
    
    config_dir = os.path.dirname(os.path.abspath(args.config))
    if args.load_path == "best_model.pth":
        args.save_path = os.path.join(config_dir, args.save_path)
        args.load_path = os.path.join(config_dir, args.load_path)

    return args

def create_dataloaders(args, train_data, val_data, test_data, max_value):
    """데이터 로더를 생성합니다."""
    transforms = []
    if args.use_augment_warp:
        transforms.append(AugmentWarp(limit=args.warp_limit))
    if args.use_random_noise:
        transforms.append(AddGaussianNoise(max_value=max_value, mean=args.noise_mean, std=args.noise_std))
    if args.use_augment_swap:
        transforms.append(SwapAugment(swap_prob=args.swap_prob))

    train_dataset = VehicleDataset(train_data, transform=transforms)
    val_dataset = VehicleDataset(val_data)
    test_dataset = VehicleDataset(test_data, transform=[SwapAugment(swap_prob=0.1)] if args.test_swap else None)
    infer_dataset = VehicleInferDataset(args, test_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, infer_loader

def create_model(args, col_dim, reg_dim, num_classes, sequence_len, intent_num):
    """설정에 따라 적절한 모델을 생성합니다."""
    model_types = {
        "lstm": lambda: SimplifiedLSTMRNN(args, args.hidden_size, args.num_layers, col_dim, reg_dim, num_classes, sequence_len, intent_num, args.model_type),
        "rnn": lambda: SimplifiedLSTMRNN(args, args.hidden_size, args.num_layers, col_dim, reg_dim, num_classes, sequence_len, intent_num, args.model_type),
        "tabgpt2": lambda: HierarchicalGPT2(args, args.hidden_size, args.num_layers, args.num_heads, col_dim, reg_dim, num_classes, sequence_len, intent_num),
        "tabllama": lambda: HierarchicalLlama(args, args.hidden_size, args.num_layers, args.num_heads, col_dim, reg_dim, num_classes, sequence_len, intent_num),
        "vanillatf": lambda: SimplifiedHierarchicalTransformer(args, args.hidden_size, args.num_layers, args.num_heads, col_dim, reg_dim, num_classes, sequence_len),
        "tabbert": lambda: HierarchicalTransformer(args, args.hidden_size, args.num_layers, args.num_heads, col_dim, reg_dim, num_classes, sequence_len)
    }
    
    if args.model_type not in model_types:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    return model_types[args.model_type]()

def main():
    """메인 실행 함수"""
    args = parse_args()
    set_seed(args.seed)

    # 데이터 전처리
    preprocessor = DataPreprocessor()
    dataframes = preprocessor.load_xml_to_dataframes(args.file_path)
    processed_data, max_value, data_dim, max_seq_len, intent_mapping, num_intent = preprocessor.prepare_data(dataframes)
    
    print("intent_mapping", intent_mapping)
    print("args", args)

    # 데이터 분할 및 증강
    if args.in_dist:
        train_data, val_data, test_data = preprocessor.augment_sequence_data_indist(processed_data)
    else: 
        train_data, val_data, test_data = preprocessor.split_data(processed_data)
        train_data = preprocessor.augment_sequence_data(train_data)
        val_data = preprocessor.augment_sequence_data(val_data)
        test_data = preprocessor.augment_sequence_data(test_data)
    
    print(f"Data sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # 모델 파라미터 설정
    col_dim = args.col_dim
    region_len = data_dim // col_dim
    sequence_len = max_seq_len
    num_classes = int(max_value + 1)

    print(f"Model parameters: col_dim={col_dim}, region_len={region_len}, sequence_len={sequence_len}, num_classes={num_classes}")

    # 데이터 로더 및 모델 생성
    train_loader, val_loader, test_loader, infer_loader = create_dataloaders(args, train_data, val_data, test_data, max_value)
    model = create_model(args, col_dim, region_len, num_classes, sequence_len, num_intent)
    
    # 모델 학습
    if args.train:
        print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
        trainer = Trainer(args, model, train_loader, val_loader, test_loader, num_epochs=args.num_epochs,
                          device=args.device, num_classes=num_classes, lr=args.learning_rate,
                          patience=args.patience, min_delta=args.min_delta)
        trained_model = trainer.train()
    
    # 모델 테스트
    if args.test:
        print("Running test and predict...")
        model.load_state_dict(torch.load(args.load_path))
        model.to(args.device)
        
        trainer = Trainer(args, model, train_loader, val_loader, test_loader, num_epochs=args.num_epochs,
                          device=args.device, num_classes=num_classes, lr=args.learning_rate,
                          patience=args.patience, min_delta=args.min_delta)
                
        results, predictions = trainer.test_and_predict(test_loader)
        
        print("Overall Test Results:")
        for key, value in results.items():
            if key != "intent_metrics":
                print(f"{key}: {value:.4f}")
        
        print("\nIntent-specific metrics:")
        for intent, metrics in results["intent_metrics"].items():
            print(f"Intent: {intent}")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

    # 모델 추론
    if args.infer:
        run_inference_bert(model=model, model_path=args.load_path, inference_dataloader=infer_loader,
                           device=args.device, sequence_len=sequence_len, initial_sequence_len=args.init_seq)

if __name__ == "__main__":
    main()
