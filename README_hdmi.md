## 목차

1. [요구 사항](#1-요구-사항)
2. [설치](#2-설치)
3. [디렉터리 구조](#3-디렉터리-구조)
4. [Hydra 설정 사용법](#4-hydra-설정-사용법)
5. [학습](#5-학습)
6. [정책 재생 / 시각화 (play.py)](#6-정책-재생--시각화-playpy)
7. [평가 (eval.py)](#7-평가-evalpy)
8. [모션 데이터 확인](#8-모션-데이터-확인)
9. [SIM-TO-REAL](#9-sim-to-real)

---

## 1. 요구 사항

### 검증된 환경

이 포크가 실제로 돌아가는 것을 확인한 조합입니다. 원본 README의 조합(IsaacSim 4.5.0 / IsaacLab v2.2.0 / Python 3.10)이 아니라 **아래 조합을 사용하세요.**

| 항목 | 버전 | 비고 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | |
| NVIDIA 드라이버 | 580.x (≥ 535) | `nvidia-smi`로 확인 |
| GPU | RTX 5000 Ada (16 GB) | VRAM에 맞춰 `task.num_envs` 조절 |
| Python | **3.11** | conda 환경 |
| Isaac Sim | **5.0.0** (pip) | `isaacsim[all,extscache]` |
| IsaacLab | **v2.3.0** | 소스 클론 + editable 설치 |
| PyTorch | **2.7.0+cu128** | Isaac Sim이 함께 설치 |
| NumPy | **1.26.x** | Isaac Sim 요구사항, 2.x 금지 |
| torchrl / tensordict | **0.7.0 / 0.7.0** | `setup.py`에 고정 |
| git-lfs | 3.x | `*.usd` 에셋이 LFS로 관리됨 |

### 사전 준비

```bash
# NVIDIA 드라이버 확인 (CUDA 12.8 런타임을 쓰므로 드라이버가 충분히 최신이어야 함)
nvidia-smi

# git-lfs 설치 (에셋 USD 파일 때문에 필수)
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
```

---

## 2. 설치

전체 설치는 **conda 환경 → Isaac Sim → IsaacLab → 이 저장소** 순서입니다.
순서를 지켜야 합니다. `pip install -e .`가 `torch==2.7.0`을 고정하기 때문에, Isaac Sim보다 먼저 설치하면 CUDA 빌드가 어긋날 수 있습니다.

### 2.1 Conda 환경

```bash
conda create -n hdmi python=3.11 -y
conda activate hdmi
pip install --upgrade pip
```

### 2.2 Isaac Sim 5.0.0

```bash
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# 설치 확인 (GUI가 뜨면 성공, 첫 실행은 셰이더 캐시 때문에 몇 분 걸림)
isaacsim
```

### 2.3 IsaacLab v2.3.0

원하는 워크스페이스 아래에 클론합니다 (이 저장소와 형제 디렉터리일 필요는 없습니다).

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0

./isaaclab.sh -i none
```

### 2.4 이 저장소 (HDMI TienKung Pro)

```bash
git clone https://github.com/cailab-hy/HDMI_Tienkung_Pro.git
cd HDMI_Tienkung_Pro

# USD 에셋은 git-lfs로 저장되어 있음 (30개 파일). 반드시 받아야 함
git lfs pull

pip install -e .
```

### 2.5 추가 패키지

`setup.py`에 빠져 있지만 코드가 실제로 쓰는 패키지들입니다.

```bash
pip install opencv-python tqdm pyzmq
```

| 패키지 | 필요한 곳 |
|---|---|
| `tqdm` | 모션 로더(`active_adaptation/utils/motion.py`), 학습 루프 — **항상 필요** |
| `opencv-python` | `camera_viewer=opencv` 카메라 창, 비디오 렌더링 (지연 import) |
| `pyzmq` | `active_adaptation/utils/liveplot.py` (선택) |
| `pyqtgraph` | 실시간 플롯(`liveplot.py`)을 쓸 때만 (선택) |

### 2.6 설치 검증

```bash
conda activate hdmi
cd /path/to/HDMI_Tienkung_Pro

# 1) 임포트 + CUDA 확인 → "2.7.0+cu128 True" 가 나와야 정상
python -c "import isaacsim, isaaclab, active_adaptation, torch; print(torch.__version__, torch.cuda.is_available())"

# 2) NumPy가 1.26.x인지 확인 (2.x면 Isaac Sim이 깨집니다)
python -c "import numpy; print(numpy.__version__)"

# 3) 레퍼런스 모션을 Isaac Sim에서 재생해 보기 (정책 없이 데이터만 확인)
python scripts/play.py algo=ppo_roa_train task=tienkung_pro/hdmi/scenario \
  +task.command.replay_motion=true headless=false task.num_envs=1
```

### 2.7 wandb 로그인 (선택)

학습 스크립트는 기본적으로 wandb에 로깅합니다(`wandb.mode: online`).

```bash
wandb login          # 로깅을 쓸 경우
# 또는 매 실행마다 wandb.mode=disabled 를 붙여 로컬로만 실행
```

### 2.8 실행 위치

**모든 스크립트는 저장소 루트에서 실행**해야 합니다 (`cfg/`, `data/` 경로가 상대 경로로 참조됩니다).

```bash
conda activate hdmi
cd /path/to/HDMI_Tienkung_Pro
```

---

## 3. 디렉터리 구조

| 경로 | 설명 |
|---|---|
| `active_adaptation/envs/` | 환경 본체. MDP 구성요소(command / observation / reward / termination / randomization)를 조합해서 task를 만듦 |
| `active_adaptation/envs/mdp/commands/hdmi/` | HDMI 핵심. 모션 트래킹 command, 관측, 보상, 종료조건 |
| `active_adaptation/learning/` | PPO 계열 알고리즘 (`ppo`, `ppo_roa`, `ppo_amp`) |
| `active_adaptation/assets/` | 로봇 URDF/메시(`tienkung_pro_collision_like_full/`)와 오브젝트 USD(`objects/`) |
| `active_adaptation/utils/` | 모션 데이터 로더, ONNX export, wandb 유틸, **평가 metric(`eval_metrics.py`)**, **Isaac UI 이미지 창(`ui_image.py`)** |
| `cfg/` | Hydra 설정. `cfg/task/`(태스크), `cfg/base/`(공용), `train.yaml`/`play.yaml`/`eval.yaml`(엔트리별) |
| `scripts/` | 실행 진입점: `train.py`, `train_sequential.py`, `play.py`, `eval.py`, `helpers.py` |
| `data/motion/` | 리타게팅된 모션 데이터 (`motion.npz` + `meta.json`) |

현재 등록된 태스크는 `cfg/task/tienkung_pro/hdmi/scenario.yaml` 하나이며, 공용 베이스는 `cfg/task/base_tienkung_pro/hdmi-base.yaml`입니다.
로봇 에셋은 `active_adaptation/assets/tienkung_pro.py`가 가리키는
`tienkung_pro_collision_like_full/tienkung_pro/urdf/…_body_capsule_like.urdf`를 사용합니다.

---

## 4. Hydra 설정 사용법

각 스크립트는 자기 전용 설정 파일을 읽습니다.

| 스크립트 | 설정 파일 | 기본값 특징 |
|---|---|---|
| `scripts/train.py` | `cfg/train.yaml` | `headless: true`, wandb 로깅 on, 출력 `outputs/` |
| `scripts/train_sequential.py` | `cfg/train_sequential.yaml` | `stages:` 목록을 순서대로(teacher → finetune) 자동 실행 |
| `scripts/play.py` | `cfg/play.yaml` | `headless: true`, `task.num_envs: 100`, 출력 `outputs_play/` |
| `scripts/eval.py` | `cfg/eval.yaml` | `headless: true`, 에피소드 단위 평가 |

> 세 설정 파일 모두 `defaults`의 task가 존재하지 않는 placeholder(`TieknungPro/TienkungProTrack-walk`)입니다.
> **항상 `task=...`를 명시적으로 넘겨야 합니다.**

**task 선택** — `cfg/task/` 아래 상대 경로(확장자 제외)를 그대로 씁니다.

```bash
task=tienkung_pro/hdmi/scenario     # cfg/task/tienkung_pro/hdmi/scenario.yaml
ls cfg/task/tienkung_pro/hdmi       # 사용 가능한 task 확인
```

**algo 선택** — 대부분 YAML이 아니라 코드의 ConfigStore에 등록되어 있습니다 (`active_adaptation/learning/ppo/*.py`의 `cs.store`).

| 이름 | 용도 |
|---|---|
| `ppo_roa_train` | teacher(특권 정보 사용) 학습 |
| `ppo_roa_adapt` / `ppo_roa_finetune` | student 적응 / 파인튜닝 |
| `ppo_roa_train_est` / `ppo_roa_adapt_est` | depth 카메라 + 추정기(estimator)를 쓰는 변형 |
| `ppo`, `ppo_amp_*`, `hier` | 기타 베이스라인 |

**override 문법**

```bash
task.num_envs=1024                # 기존 키 덮어쓰기
+task.command.replay_motion=true  # 없는 키 추가는 '+'
task.viewer.eye=[6.4,5.0,4.7]     # 리스트는 공백 없이
wandb.mode=disabled               # 로컬 디버깅
```

---

## 5. 학습

```bash
# teacher 학습 student 파인튜닝
python scripts/train_sequential.py  stages=[ppo_roa_train,ppo_roa_adapt,ppo_roa_finetune] task=tienkung_pro/hdmi/scenario task.num_envs=4096 

# 이전 스테이지의 wandb run 을 다음 스테이지 checkpoint 로 자동 연결)
python scripts/train_sequential.py task=tienkung_pro/hdmi/scenario  stages=[ppo_roa_train_est,ppo_roa_adapt_est] task.num_envs=2048 app.enable_cameras=true
```

자주 쓰는 옵션: `headless=false`(창 띄우기), `task.num_envs=2048`(GPU 메모리에 맞게 조절), `save_interval=100`, `seed=0`, `resume_training=true`(옵티마이저 상태까지 복원).

`checkpoint_path`는 세 가지 형식을 지원합니다 (`active_adaptation/utils/wandb.py`의 `parse_checkpoint_path`).

```
/abs/path/checkpoint_final.pt       # 로컬 파일
run:<entity/project/run_id>         # wandb에서 최신 체크포인트 다운로드
run:<entity/project/run_id>:2200    # 특정 스텝 체크포인트
```

---

## 6. 정책 재생 / 시각화 (`play.py`)

```bash
python scripts/play.py algo=ppo_roa_adapt_est task=tienkung_pro/hdmi/scenario \
  checkpoint_path=/abs/path/checkpoint_final.pt \
  headless=false task.num_envs=1 app.enable_cameras=true export_policy=true
```

`cfg/play.yaml`의 기본값은 `headless: true`, `task.num_envs: 100`입니다. 눈으로 보려면 위처럼 `headless=false`를, 뷰어 카메라 좌표를 그대로 쓰려면 `task.num_envs=1`을 함께 넘기세요.

**뷰어 카메라** — task 설정의 `viewer` 항목으로 지정합니다. `task.num_envs=1`이면 env 원점이 월드 원점과 같으므로, 아래 값이 곧 월드 좌표입니다.

```bash
task.viewer.eye=[6.4,5.0,4.7] task.viewer.lookat=[0.,0.,0.5]
```

**로봇 1인칭 카메라** — 로봇 머리에 붙은 `tiled_camera` 영상을 볼 수 있습니다 (`cfg/play.yaml`).

| 키 | 값 | 설명 |
|---|---|---|
| `visualize_camera` | `true`/`false` | 카메라 영상 표시 여부 |
| `camera_mode` | `ego_rgb` / `ego_depth` | RGB 또는 깊이(JET 컬러맵) |
| `camera_viewer` | `isaac` / `opencv` | `isaac`: IsaacLab UI 안의 도킹 가능한 창(기본), `opencv`: 별도 cv2 창 |

---

## 7. 평가 (`eval.py`)

정해진 개수의 **에피소드가 실제로 끝날 때까지** 롤아웃하면서 트래킹 오차와 성공률을 집계합니다. 학습은 하지 않고, 행동은 결정론적(`ExplorationType.MODE`)입니다.

```bash
python scripts/eval.py --task tienkung_pro/hdmi/scenario \
  --num_envs 200 --num_episodes 1000 --headless \
  --algo ppo_roa_adapt_est \
  --checkpoint /abs/path/checkpoint_final.pt
```

| 플래그 | 대응 hydra 키 | 설명 |
|---|---|---|
| `--task` | `task` | 태스크 설정 경로 |
| `--num_envs` | `task.num_envs` | 병렬 환경 수 |
| `--num_episodes` | `num_episodes` | 통계에 포함할 총 에피소드 수 (기본 1000) |
| `--algo` | `algo` | 체크포인트와 반드시 일치해야 함 |
| `--checkpoint` | `checkpoint_path` | 로컬 경로 또는 `run:` 형식 |
| `--headless` / `--no-headless` | `headless` | 창 없이 실행 |
| `--seed` | `seed` | 랜덤 시드 |
| `--max_episode_length` | `task.max_episode_length` | 에피소드 최대 길이 |
| `--stochastic` | `deterministic=false` | 행동을 샘플링 (기본은 결정론적) |
| `--no_print_success` | `print_success_episodes=false` | 성공 trial 목록 출력 끄기 (기본은 출력) |
| `--render` | `eval_render=true` | 기존 고정 길이 롤아웃 + 비디오 녹화 경로 |

`key=value` 형태의 일반 hydra override도 함께 쓸 수 있습니다 (예: `task.max_episode_length=600`, `progress_every=100`).

**동작 방식**

- 환경마다 종료 시점이 다르므로 **환경별로 독립적으로** 에피소드 종료를 추적하고, 종료된 에피소드만 하나씩 집계합니다.
- 목표 개수를 넘겨서 동시에 끝난 에피소드는 버려서, 통계에는 **정확히 `--num_episodes`개**만 반영됩니다.
- 진행 상황은 `Evaluation Progress: 200 / 1000` 형태로 출력됩니다.

**출력 지표** (정의는 학습에서 쓰는 보상 항과 동일한 것을 재사용합니다)

| 지표 | 정의 | 출처 |
|---|---|---|
| Joint Position Tracking Error | `\|joint_pos - ref_joint_pos\|`의 평균/RMSE, 단위 rad | `joint_pos_error` (rewards.py) |
| Body Position Tracking Error | root(yaw만, 높이 0) 기준 **local frame**에서의 body 위치 L2 오차, 단위 m | `keypoint_pos_error_local` (rewards.py) |
| Task Success | `command_manager.success` = 레퍼런스 모션 끝까지 생존 | `RobotTracking.success`, `_Env._compute_reward` |

body 오차를 local frame에서 재는 이유는 tracking 보상이 그렇게 정의되어 있고, 월드 좌표 drift 때문에 생기는 무의미한 오차를 배제하기 위해서입니다. 실행 시작 시 어떤 joint/body 집합과 frame을 썼는지 콘솔에 출력됩니다.

**출력 예시**

```text
============================================================
                    Evaluation Results
============================================================
Total Episodes                  : 1000
Successful Episodes             : 812
Success Rate                    : 81.20 %
Mean Episode Length             : 641.3 steps

Joint Position Tracking Error
  Mean                          : 0.0840 rad
  RMSE                          : 0.1120 rad
  Joints / Samples              : 21 / 13446300

Body Position Tracking Error
  Mean                          : 0.0430 m
  RMSE                          : 0.0610 m
  Frame                         : local
  Bodies / Samples              : 10 / 6403000
============================================================
```

**실패 원인 분석**

summary 다음에, 실패한 에피소드를 종료시킨 termination 조건별 집계가 출력됩니다 (`stats["termination"]` 플래그 기준. 여러 조건이 동시에 걸릴 수 있어 합이 100%를 넘을 수 있고, 플래그가 없는 실패는 `max_episode_length` 초과입니다).

```text
============================================================
       Failure Breakdown  (188 / 1000 episodes failed)
============================================================
  cum_object_pos_error                :   121  ( 64.4% of failures)
  cum_lost_contact_steps              :    77  ( 41.0% of failures)
  cum_body_pos_error_local            :    23  ( 12.2% of failures)

  mean length of failed episodes      : 214.5 steps
  mean length of successful episodes  : 1108.0 steps
============================================================
```

**성공한 trial 목록**

summary 뒤에 성공한 에피소드만 `<env_id>:<episode_index>` 형식으로 출력됩니다 (`env_id` = 몇 번째 병렬 환경, `episode_index` = 그 환경의 몇 번째 에피소드).

```text
============================================================
       Successful Trials  (812 / 1000)
============================================================
seed = 100   |   entry format = <env_id>:<episode_index>

  0:0    0:1    0:3    1:0    1:1    1:2    1:3    1:4    2:2    3:0
  3:1    3:2    ...

env ids with >=1 success (186): [0, 1, 2, 3, ...]
env ids with all episodes successful (91): [1, 7, 12, ...]
```

끄려면 `--no_print_success` 를 붙이세요.

결과는 `scripts/eval/<task-name>/<task-name>-<MM-DD_HH-MM>.yaml`로도 저장되며, 위 목록은 `success_episodes`, `success_env_ids`, `all_success_env_ids` 키로 함께 기록됩니다.

> 참고: 평가 모드에서는 기존 환경 코드가 에피소드 시작 시점을 모션 처음(`t=0`)으로 고정하고 리셋 노이즈를 끕니다. 에피소드 간 차이는 도메인 랜덤화(질량/마찰/무게중심)에서만 발생합니다.

---

## 8. 모션 데이터 확인

모션은 `data/motion/<robot>/<task>/`에 `motion.npz` + `meta.json` 형태로 들어갑니다
(현재: `data/motion/tienkung_pro/scenario/`. 포맷은 아래 [Data Preparation](#data-preparation) 참조).

```bash
# Isaac Sim에서 레퍼런스 모션 재생
python scripts/play.py algo=ppo_roa_train task=tienkung_pro/hdmi/scenario \
  +task.command.replay_motion=true headless=false task.num_envs=1

# MuJoCo 뷰어로 확인 (터미널 2개)
python scripts/vis/mujoco_mocap_viewer.py
python scripts/vis/motion_data_publisher.py <모션-폴더-경로>
```

---

## 9. sim-to-real 

해당 package를 실제 환경에서 돌리기 위해 아래 명시된 링크를 따라가세요

https://github.com/cailab-hy/Deploy_Tienkung_Pro
