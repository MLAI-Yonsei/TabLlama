# TabLlama


## Introduction




Python 버전 3.8.0을 사용합니다.
2\. 소스코드를 클론하거나 다운로드하는 방법을 포함시킵니다.
3\. 설정 파일이 필요한 경우, 설정 파일을 준비하는 방법을 안내합니다.

```bash
# 환경설치
pip install -r requirements.txt
```



새로 학습을 진행하고 싶다면, 필요한 설정을 `.yaml` 파일에서 직접 수정하면 됩니다.

### 사용 예시 1

```bash
c
```

cd emforce77/src; conda activate pret; python main.py --config ./configs/default/config\_default.yaml이 명령어는 `emforce77/src` 디렉토리로 이동한 후, `pret`라는 Conda 환경을 활성화하고 `main.py`를 실행합니다. 설정 파일로는 `./configs/default/config_default.yaml`을 사용합니다.

### 사용 예시 2

```bash
cd emforce77/src; conda activate pret; python main.py --config ./configs/indist/config_indist.yaml
```

이 명령어는 `indist` 설정 파일(`./configs/indist/config_indist.yaml`)을 사용하여 `main.py`를 실행합니다. 앞의 명령어와 마찬가지로 `emforce77/src` 디렉토리로 이동하고, `pret` 환경을 활성화합니다.

### 사용 예시 3

```bash
cd emforce77/src; conda activate pret; python main.py --config ./configs/indist_testswap/config_indist_testswap.yaml
```

이 명령어는 `indist_testswap` 설정 파일(`./configs/indist_testswap/config_indist_testswap.yaml`)을 사용하여 `main.py`를 실행합니다. 프로젝트의 설정을 변경하여 다른 실험을 수행할 때 유용합니다.

### 사용 예시 4

```bash
cd emforce77/src; conda activate pret; python main.py --config ./configs/testswap/config_testswap.yaml
```

이 명령어는 `testswap` 설정 파일(`./configs/testswap/config_testswap.yaml`)을 사용하여 `main.py`를 실행합니다. 다른 설정을 통해 테스트 스왑 작업을 수행할 수 있습니다.

### 사용 예시 5

```bash
cd emforce77/src; conda activate pret; bash infers.sh
```

이 명령어는 `emforce77/src` 디렉토리로 이동한 후, `pret` 환경을 활성화하고 `infers.sh` 스크립트를 실행합니다. 이 스크립트는 모델 추론과 관련된 작업을 수행할 수 있습니다.

### 사용 예시 6

```bash
cd emforce77/src; conda activate pret; bash intent_result.sh
```

이 명령어는 `intent_result.sh` 스크립트를 실행하여 의도 분석과 관련된 결과를 생성합니다. 이 역시 `pret` 환경에서 실행됩니다.

## 추가 정보

inference 결과와 intent 별 성능을 확인할려면 다음 파일을 참고하면 됩니다.
`/home/emforce77/src/plot.ipynb`

데이터 생성을 원한다면 다음 코드를 실행하면 됩니다.
`/home/emforce77/src/scenario_generation.ipynb`

## 기여 방법

이 프로젝트에 기여하고 싶다면, 기여 방법에 대해 간단하게 설명하세요. 예를 들어, 포크하고 PR을 보내는 절차 등입니다.

## 라이센스

프로젝트의 라이센스 정보를 여기에 포함합니다. 예: MIT 라이센스.

## 문의

프로젝트 관련 문의 사항이나 이슈를 보고할 이메일 주소나 연락처 정보를 작성하세요.
