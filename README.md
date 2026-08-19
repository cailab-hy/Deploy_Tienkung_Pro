# big_wave_Deploy — HDMI 정책 Sim-to-Real 배포

Tienkung Pro 휴머노이드 로봇에 HDMI 프레임워크(IsaacLab + PyTorch)로 학습한 강화학습 정책을 배포하기 위한 ROS2 프로젝트입니다.
IsaacLab에서 학습된 정책(ONNX)을 실제 로봇에서 50Hz로 추론하고, 400Hz PD 제어로 모터를 구동합니다.


- **Data Transform repo**: [big_wave_GMR](https://github.com/cailab-hy/big_wave_GMR) — 참조 모션 파일
- **Train repo**: [big_wave_HDMI](https://github.com/cailab-hy/big_wave_HDMI) — 정책 학습 / 평가 / ONNX export
- **Deploy repo**: [big_wave_Deloy](https://github.com/cailab-hy/big_wave_Deploy) — Sim-to-Real 

---

## 목차

1. [요구 사항](#1-요구-사항)
2. [설치 및 환경 세팅](#2-설치-및-환경-세팅)
3. [실행 방법 (Sim-to-Real)](#3-실행-방법-sim-to-real)
4. [조이스틱 조작 및 FSM](#4-조이스틱-조작-및-fsm)
5. [정책 교체 방법](#5-정책-교체-방법)
6. [안전장치](#6-안전장치)
7. [기타](#7-기타)

---

## 1. 요구 사항

### 1.1 공통
- Ubuntu 22.04 LTS
- ROS2 Humble
- C++17 컴파일러, CMake 3.8+

### 1.2 Main PC (`rl_control_new`, `hdmi_inference`)
- Eigen3, yaml-cpp, spdlog, fmt
- OpenCV (깊이 영상 리사이즈)
- zlib (motion.npz 파싱)
- **ONNX Runtime C++** (정책 추론, `hdmi_inference` 빌드에 필수)
- bodyctrl_msgs (로봇 벤더 제공 모터 제어 메시지)

### 1.3 Orin (카메라)
- Orbbec Camera ROS2 패키지 ([OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2))

---

## 2. 설치 및 환경 세팅

### 2.1 공통 의존성 설치 (Robot Main PC + Orin)

Robot Main PC와 Orin에서 아래를 실행하여 의존성을 설치합니다.

```bash
sudo apt update
sudo apt install -y \
  libeigen3-dev libyaml-cpp-dev libspdlog-dev libfmt-dev \
  libopencv-dev zlib1g-dev
```

### 2.2 Robot Main PC 설치 항목들

실행(`rl_control_new`, `hdmi_inference`)이 모두 Robot Main PC에서 이루어지기 때문에 Orin 설치 항목과 다릅니다. 

#### 2.2.1 ONNX Runtime 설치 (Robot Main PC)

`hdmi_inference`는 ONNX Runtime C++ API를 사용합니다. [릴리스 페이지](https://github.com/microsoft/onnxruntime/releases)에서 플랫폼에 맞는 tgz를 받아 `/usr/local`에 설치합니다.

```bash
# 예시: x86_64 GPU 버전 (버전은 환경에 맞게 선택)
tar -xzf onnxruntime-linux-x64-gpu-<version>.tgz
sudo cp -r onnxruntime-linux-x64-gpu-<version>/include/* /usr/local/include/
sudo cp -r onnxruntime-linux-x64-gpu-<version>/lib/*     /usr/local/lib/
sudo ldconfig
```

> 설치 경로가 `/usr/local`이 아니면 빌드 시 `-DONNXRUNTIME_ROOT_DIR=<경로>`를 넘겨주세요.
> Jetson(aarch64)에서 빌드할 경우 aarch64용 빌드를 사용해야 합니다.

### 2.2.2 저장소 클론 

```bash
mkdir -p ~/tk_ws/src
cd ~/tk_ws/src
git cloen hhtps://github.com/cailab-hy/big_wave_Deploy.git
```

### 2.2.3 ROS2 빌드

로봇 벤더 워크스페이스(`ros2ws`, bodyctrl_msgs 등 포함)를 먼저 source 해야 `rl_control_new`가 빌드됩니다.

```bash
cd ~/tk_ws
source /opt/ros/humble/setup.bash
source ~/ros2ws/install/setup.bash        # 벤더 패키지 (bodyctrl_msgs 등)

colcon build --packages-select x_humanoid_rl_sdk rl_control_new hdmi_inference
source install/setup.bash
```

특정 패키지만 다시 빌드할 때:

```bash
colcon build --packages-select hdmi_inference
```

### 2.3. Orin 설치 항목 

깊이 카메라 패키지([OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2))만 별도 워크스페이스(예: `~/orbbec_camera_ros2`)에 설치하면 됩니다.

### 2.4 네트워크 설정

Main PC와 Orin이 같은 ROS2 네트워크에 있어야 토픽이 오갑니다.

- 두 머신의 `ROS_DOMAIN_ID`를 동일하게 설정
- 접속 정보 — IP와 계정은 **각자 휴머노이드 환경에 맞게** 사용하세요:

| 머신 | 접속 (예시) |
|---|---|
| Main PC | `ssh <user>@<MAIN_PC_IP>` |
| Orin | `ssh <user>@<ORIN_IP>` |

---

## 3. 실행 방법 (Sim-to-Real)


터미널 4개를 사용합니다. 순서대로 실행하세요.
아래 `<MAIN_PC_IP>`, `<ORIN_IP>`와 계정명은 각자 휴머노이드 환경에 맞게 바꿔서 사용하세요.

### 3.1 [Main PC 터미널 1] body_control — 모터 구동

```bash
ssh <user>@<MAIN_PC_IP>
sudo su
source ros2ws/install/setup.bash
sudo systemctl stop proc_manager.service   # 기본 프로세스 매니저 중지
ros2 launch body_control body.launch.py
```

### 3.2 [Main PC 터미널 2] rl_control_new — 400Hz 하드웨어 제어

```bash
ssh <user>@<MAIN_PC_IP>
sudo su
source ros2ws/install/setup.bash
source tk_ws/install/setup.bash
ros2 launch rl_control_new rl.launch.py
```

### 3.3 [Main PC 터미널 3] hdmi_inference — 50Hz 정책 추론

```bash
ssh <user>@<MAIN_PC_IP>
source /opt/ros/humble/setup.bash
source ~/tk_ws/install/setup.bash
ros2 launch hdmi_inference hdmi_inference.launch.py
```

### 3.4 [Orin 터미널 1] 깊이 카메라

```bash
ssh <user>@<ORIN_IP>
cd ~/orbbec_camera_ros2
source install/setup.bash
ros2 launch orbbec_camera gemini_330_series.launch.py
```

모든 노드가 뜨면 `hdmi_inference` 로그에 5초마다 상태가 출력됩니다:

```
State=0, Policy=0, q_d[0]=..., waist_reset=0, depth=19200(0.30~4.50m)
```

`depth=19200`이 확인되면 카메라 연결 정상, `State`/`Policy`는 아래 FSM 상태를 나타냅니다.

---

## 4. 조이스틱 조작 및 FSM

FSM은 `hdmi_inference`에서 관리하며, 조이스틱(`/sbus_data`)으로 상태를 전환합니다.

```
        [D / X]           [A+G / A]          [C / Y]
STOP ──────────▶ ZERO ──────────────▶ MLP ──────────▶ STOP
 현재 위치 유지    63스텝(1.26초) 보간    50Hz 정책 추론
                 → 초기 자세            
```

| 순서 | 키 (Yunzhuo / Xbox) | 상태 | 동작 |
|---|---|---|---|
| 시작 | - | **STOP** | 모든 모터 현재 위치 유지 |
| 1 | **D** / **X** | **ZERO** | 1.26초에 걸쳐 초기 자세로 보간 이동 |
| 2 | **A+G** / **A** | **MLP** | ONNX 정책 실행 시작 |
| 언제든 | **C** / **Y** | **STOP** | 즉시 정지 (현재 위치 유지) |

---

## 5. 정책 교체 방법

학습 저장소에서 export한 정책 파일 세트를 config 폴더에 넣고 노드 코드에서 지정합니다.

1. 학습 저장소(big_wave_HDMI)에서 `play.py ... export_policy=true`로 ONNX export
   (자세한 방법은 [학습 저장소](https://github.com/cailab-hy/big_wave_HDMI) README 참조)
2. 정책 폴더 구성 — 4개 파일 세트가 필요합니다:

   ```
   hdmi_inference/config/policy_<이름>/
   ├── policy-<run_id>.onnx     # ONNX 정책
   ├── policy-<run_id>.yaml     # 관절 순서, action scale, PD gain 등 설정
   ├── policy-<run_id>.json     # 메타 정보
   ├── motion.npz               # 레퍼런스 모션 데이터
   └── meta.json                # 모션 메타 정보
   ```

3. [hdmi_inference_node.cpp](./hdmi_inference/src/hdmi_inference_node.cpp)의 생성자에서 정책 디렉터리/이름 수정:

   ```cpp
   std::string stand1_dir  = pkg_path + "/config/policy_stand1";
   std::string stand1_name = "policy-b3tvcz86-final";   // 확장자 제외
   ```

4. 다시 빌드: `colcon build --packages-select hdmi_inference`

---

## 6. 안전장치

| 항목 | 동작 |
|---|---|
| **Action 타임아웃 (150ms)** | rl_control_new가 150ms 이상 action을 못 받으면 자동 STOP (현재 위치 유지, 속도/토크 0) |
| **PD gain fallback** | hdmi_inference 연결 전에는 rl_control_new가 로컬 policy YAML의 PD gain 사용, `/hdmi/pd_gains` 수신 시 자동 갱신 |
| **Waist reset** | STOP→ZERO 전환 시 waist 모터 리셋 플래그를 action에 포함하여 Main PC가 처리 |
| **즉시 정지** | 조이스틱 C/Y 버튼으로 언제든 STOP 전환 가능 |

---

## 7. 기타

자세한 Deploy 정보는 문서를 참고하세요

---

## 관련 문서

- [rl_control_new/README_HDMI.md](./rl_control_new/README_HDMI.md) — 2-머신 아키텍처, 토픽 포맷, 관절 매핑 상세
- [rl_control_new/README.md](./rl_control_new/README.md) — rl_control_new 패키지 문서 (벤더 원본)
- [x_humanoid_rl_sdk/README.md](./x_humanoid_rl_sdk/README.md) — 로봇 SDK 문서 (벤더 원본)

---

본 프로젝트는 빅웨이브와 한양대학교 박태준 교수님 연구실 CAILAB에서 실시한 산학과제 결과물입니다.
