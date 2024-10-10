# TabLlama


## Introduction

In an increasingly complex world, accurate prediction of multi-variate behaviors across domains like aviation, economics, and healthcare has become essential. Traditional models excel at univariate predictions but struggle with capturing the intricate interdependencies inherent in multi-variate data. This paper proposes a hierarchical transformer model designed for multi-variate time series prediction, specifically focused on predicting vehicle behavior. Leveraging the self-attention mechanism, the model captures both long-range and detailed temporal relationships in multi-variate scenarios, while the hierarchical structure facilitates the integration of variable-level information from tabular data. Our approach incorporates novel data augmentation techniques such as noise injection, sequence warping, and swapping to improve model robustness under varied real-world conditions. Experimental results demonstrate that the proposed transformer-based model significantly outperforms conventional baselines, particularly in handling complex multi-variate data. Data augmentation consistently improved prediction performance across multiple metrics, underscoring the model's capability to generalize better under sparse and noisy conditions. The model's superior performance in prediction tasks highlights its potential for enhancing predictive capabilities in aviation and other domains requiring sophisticated multi-variate analysis. By integrating advanced transformer architectures with domain-specific data augmentation, this work addresses a critical gap in current predictive modeling, offering a powerful tool for anticipating complex behaviors in dynamic environments.


Python==3.8.0

```bash
# install envrioment
pip install -r requirements.txt
```



새로 학습을 진행하고 싶다면, 각 task별 config '.yaml` 파일에서 train: True 로 설정하시면 됩니다.
기타 하이퍼파라미터 역시 수정가능합니다.

### Out-Sequence Task

```bash
python main.py --config ./configs/default/config\_default.yaml
```

### In-Sequence Task

```bash
python main.py --config ./configs/indist/config_indist.yaml
```

### Out-Sequence with swapped test dataset Task

```bash
python main.py --config ./configs/testswap/config_testswap.yaml 
```

### In-Sequence with swapped test dataset Task

```bash
python main.py --config ./configs/indist_testswap/config_indist_testswap.yaml
```

### Autoregressive Inference with 1,2,3,4 initial sequences

```bash
bash infers.sh
```


### Reporting Metrics of each Intent

```bash
bash intent_result.sh
```


## 추가 정보

Autoregressive Inference 결과와 Intent 별 성능을 확인할려면 다음 ipynb 파일을 실행하면 됩니다.
`/home/emforce77/src/plot.ipynb`

데이터 생성을 원한다면 다음 코드를 실행하면 됩니다.
`/home/emforce77/src/scenario_generation.ipynb`

