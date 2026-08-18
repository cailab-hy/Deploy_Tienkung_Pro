# HDMI Policy Deployment for Tienkung Pro

HDMI 프레임워크(PyTorch + TorchRL + Isaac Lab)로 학습된 강화학습 정책을 Tienkung Pro 휴머노이드 로봇에 배포하기 위한 C++ ROS2 구현.

## 아키텍처 (2-머신 분리)

```
┌─────────────────────────────────────────────────────────────────┐
│                  같은 ROS2 네트워크 (같은 domain_id)             │
│                                                                 │
│  Main PC                          Jetson Orin (GPU)             │
│  ┌───────────────────┐           ┌───────────────────┐         │
│  │ [rl_control_new]  │           │ [hdmi_inference]   │         │
│  │                   │  50Hz     │                   │         │
│  │ 400Hz 모터 제어    │◄────────►│ 50Hz 정책 추론     │         │
│  │ 센서 읽기          │ ROS2     │ ONNX Runtime(GPU) │         │
│  │ S/P 변환          │  토픽     │ 깊이 카메라(로컬)   │         │
│  │ PD 제어           │           │ FSM 상태 관리      │         │
│  └───────────────────┘           └───────────────────┘         │
│                                                                 │
│  /sbus_data (조이스틱) ── 두 머신 모두 구독 가능 ──              │
└─────────────────────────────────────────────────────────────────┘
```

### Main PC 역할 (하드웨어 제어, `rl_control_new` 패키지)
- **400Hz**: 센서 읽기, S/P 변환, HW↔Isaac 매핑, PD 모터 제어
- **50Hz**: sensor_state를 Orin에 발행 (매 8번째 스텝)
- **수신**: Orin에서 온 action을 적용 (desired_pos/vel/tor + FSM state)
- **안전장치**: 150ms 타임아웃 → 자동 STOP (현재 위치 유지)
- **ONNX Runtime 불필요** (GPU 없음)

### Jetson Orin 역할 (두뇌, `hdmi_inference` 패키지)
- **50Hz**: 정책 추론 (ONNX Runtime + CUDA)
- **입력**: sensor_state(Main PC), 깊이 카메라(로컬), 조이스틱
- **처리**: FSM(STOP/ZERO/MLP) + ObservationBuilder + MotionData + OnnxInference
- **출력**: action(desired_pos/vel/tor + FSM state + flags) → Main PC

---

## ROS2 토픽

### `/hdmi/sensor_state` (Main PC → Orin, 50Hz)
`std_msgs/msg/Float64MultiArray` [39 values]

| Index | 내용 | 차원 |
|-------|------|------|
| 0-29 | joint_pos (isaac 순서, S/P 변환 후) | 30 |
| 30-32 | IMU euler (yaw, pitch, roll) | 3 |
| 33-35 | IMU angular_vel (x, y, z) | 3 |
| 36-38 | IMU linear_accel (x, y, z) | 3 |

### `/hdmi/action` (Orin → Main PC, 50Hz)
`std_msgs/msg/Float64MultiArray` [93 values]

| Index | 내용 | 차원 |
|-------|------|------|
| 0-29 | desired_pos (isaac 순서) | 30 |
| 30-59 | desired_vel (isaac 순서) | 30 |
| 60-89 | desired_tor (isaac 순서) | 30 |
| 90 | fsm_state (0=STOP, 1=ZERO, 2=MLP) | 1 |
| 91 | disable_joints (0 or 1) | 1 |
| 92 | waist_reset_requested (0 or 1) | 1 |

### `/hdmi/pd_gains` (Orin → Main PC, 1회, transient_local)
`std_msgs/msg/Float64MultiArray` [60 values]

| Index | 내용 | 차원 |
|-------|------|------|
| 0-29 | kp (isaac 순서) | 30 |
| 30-59 | kd (isaac 순서) | 30 |

---

## 데이터 흐름

```
Main PC (400Hz)                    Orin (50Hz)
──────────────                     ──────────
센서 읽기 (관절/IMU)
  │
S/P 변환 (병렬→직렬)
  │
HW → Isaac 매핑
  │
[매 8번째 스텝]
sensor_state 발행 ──────────────► 수신
  │                                │
  │                                ObservationBuilder.update()
  │                                MotionData.step() → command[376]
  │                                buildPolicyObs() → policy[276]
  │                                깊이 카메라 → depth[19200]
  │                                OnnxInference.run() → action[30]
  │                                FSM 처리
  │                                │
수신 ◄──────────────────────────── action 발행
  │
Isaac → HW 매핑
  │
S/P 역변환 (직렬→병렬)
  │
모터 커맨드 발행
```

### 제어 주파수
- **Main PC 외부 루프**: 400Hz (2.5ms) - 센서 읽기, PD 제어, 모터 커맨드
- **센서 발행**: 50Hz (매 8번째 400Hz 스텝)
- **Orin 정책 추론**: 50Hz (20ms 타이머)
- 네트워크 부하: ~20 KB/s (무시 가능)

---

## 안전장치

### Action 타임아웃 (150ms)
Main PC가 150ms 이상 Orin에서 action을 받지 못하면 자동 STOP:
- 현재 위치 유지 (q_d = q_actual)
- 속도/토크 = 0
- 로그 경고 출력

### PD Gains Fallback
Orin 연결 전까지 Main PC는 로컬 policy YAML에서 PD gain을 로드하여 사용.
Orin이 `/hdmi/pd_gains`를 발행하면 자동 업데이트.

### Waist Reset
STOP → ZERO 전환 시 waist 모터 리셋이 필요.
Orin에서 전환을 감지 → action 메시지에 `waist_reset_requested=1` 플래그 포함 → Main PC에서 waist position 커맨드 발행.

---

## 파일 구조

### `hdmi_inference/` (Orin 패키지, 신규)
```
hdmi_inference/
├── CMakeLists.txt
├── package.xml
├── config/policy_kbc/           # 정책 파일
│   ├── policy-fmtq3g5b-final.onnx/json/yaml
│   ├── motion.npz
│   └── meta.json
├── launch/hdmi_inference.launch.py
├── include/
│   ├── HdmiPolicyRunner.h       # 50Hz 적응 (ZERO_DURATION=63)
│   ├── OnnxInference.h
│   ├── ObservationBuilder.h
│   ├── MotionData.h
│   └── Joystick.h
├── src/
│   ├── hdmi_inference_node.cpp  # ROS2 노드 (메인)
│   ├── HdmiPolicyRunner.cpp
│   ├── OnnxInference.cpp
│   ├── ObservationBuilder.cpp
│   ├── MotionData.cpp
│   └── Joystick.cpp
└── third_party/
    ├── cnpy.h/cpp
    └── nlohmann/json.hpp
```

### `rl_control_new/` (Main PC 패키지, 수정)
```
rl_control_new/
├── src/plugins/rl_control_new/
│   ├── include/RLControlNewPlugin.h   # Orin 통신 멤버 추가
│   └── src/RLControlNewPlugin.cpp     # sensor_state 발행 + action 수신
├── CMakeLists.txt                     # ONNX Runtime 의존성 제거
└── config/policy_kbc/                 # PD gain fallback용
```

---

## 관측 구조 상세

### Command 관측 [376 dims] - MotionData에서 계산
| 구성 요소 | 차원 | 설명 |
|-----------|------|------|
| ref_body_pos_future_local | 5 steps x 17 bodies x 3 = **255** | 루트 프레임 기준 미래 바디 위치 |
| ref_joint_pos_future | 5 steps x 24 joints = **120** | 미래 관절 위치 레퍼런스 |
| ref_motion_phase | **1** | 모션 페이즈 `(t % 487) / 487` |
| **합계** | **376** | |

### Policy 관측 [276 dims] - ObservationBuilder에서 계산
| 구성 요소 | 차원 | 설명 |
|-----------|------|------|
| root_ang_vel (step=0) | **3** | IMU 각속도 (x, y, z) |
| projected_gravity (step=0) | **3** | IMU quat로 계산한 중력 방향 |
| joint_pos_history (6 steps) | 30 x 6 = **180** | 관절 위치 히스토리 (steps=[0,1,2,3,4,8]) |
| prev_actions (3 steps) | 30 x 3 = **90** | 이전 3스텝 행동 버퍼 |
| **합계** | **276** | |

### Depth 관측 [1 x 120 x 160 = 19200]
- ROS 토픽: `/camera/depth/image_raw` (Orin에서만 로컬 구독)
- 지원 인코딩: 32FC1 (float m), 16UC1 (uint16 mm → float m 변환)

---

## 관절 순서 매핑

### HW 순서 (bodyIdMap, 30개)
```
[0-5]:   L leg (hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll)
[6-11]:  R leg (hip_roll, hip_pitch, hip_yaw, knee, ankle_pitch, ankle_roll)
[12]:    waist_yaw
[13-15]: head (roll, pitch, yaw)
[16-22]: L arm (shoulder_pitch/roll/yaw, elbow, wrist_yaw/pitch/roll)
[23-29]: R arm (shoulder_pitch/roll/yaw, elbow, wrist_yaw/pitch/roll)
```

### Isaac 순서 (정책, 30개)
```
[0]: body_yaw → hw[12]     [15]: shoulder_yaw_r → hw[25]
[1]: hip_roll_l → hw[0]    [16]: knee_pitch_l → hw[3]
[2]: hip_roll_r → hw[6]    [17]: knee_pitch_r → hw[9]
[3]: head_yaw → hw[15]     [18]: elbow_pitch_l → hw[19]
[4]: shoulder_pitch_l → hw[16]  [19]: elbow_pitch_r → hw[26]
[5]: shoulder_pitch_r → hw[23]  [20]: ankle_pitch_l → hw[4]
[6]: hip_pitch_l → hw[1]   [21]: ankle_pitch_r → hw[10]
[7]: hip_pitch_r → hw[7]   [22]: elbow_yaw_l → hw[20]
[8]: head_pitch → hw[14]   [23]: elbow_yaw_r → hw[27]
[9]: shoulder_roll_l → hw[17]   [24]: ankle_roll_l → hw[5]
[10]: shoulder_roll_r → hw[24]  [25]: ankle_roll_r → hw[11]
[11]: hip_yaw_l → hw[2]    [26]: wrist_pitch_l → hw[21]
[12]: hip_yaw_r → hw[8]    [27]: wrist_pitch_r → hw[28]
[13]: head_roll → hw[13]   [28]: wrist_roll_l → hw[22]
[14]: shoulder_yaw_l → hw[18]  [29]: wrist_roll_r → hw[29]
```

매핑 배열: Main PC에서는 `RLControlNewPlugin::HW_TO_ISAAC[30]`/`ISAAC_TO_HW[30]`, Orin에서는 `HdmiPolicyRunner::HW_TO_ISAAC[30]`/`ISAAC_TO_HW[30]`.

---

## FSM 상태 전이

```
        [D / X]           [A+G / A]          [C / Y]
STOP ──────────▶ ZERO ──────────────▶ MLP ──────────▶ STOP
 │                │                    │
 │  현재 위치     │  63스텝 보간        │  50Hz 정책 추론
 │  유지          │  (1.25초)          │  action_scale=0.25
 │                │  → default pos     │  action ∈ [-100, 100]
```

- **STOP**: 모든 모터 현재 위치 유지, 정책 비실행
- **ZERO**: 현재 위치 → `default_joint_pos`로 63스텝(50Hz × 1.25초)에 걸쳐 선형 보간
- **MLP**: ONNX 정책 추론 실행, `desired_pos = default_joint_pos + action * action_scale`

FSM은 **Orin (hdmi_inference)** 에서 관리. Main PC는 Orin이 보내는 FSM 상태에 따라 동작.

---

## 의존성

### Main PC (`rl_control_new`)
| 라이브러리 | 용도 |
|-----------|------|
| rclcpp, std_msgs, sensor_msgs | ROS2 통신 |
| bodyctrl_msgs | 모터 제어 메시지 |
| yaml-cpp | PD gain fallback 로딩 |
| Eigen3 | 행렬 연산 |
| funcSPTrans | S/P 변환 |

**ONNX Runtime 불필요** (정책 추론 없음)

### Jetson Orin (`hdmi_inference`)
| 라이브러리 | 용도 |
|-----------|------|
| rclcpp, std_msgs, sensor_msgs | ROS2 통신 |
| ONNX Runtime C++ (CUDA) | ONNX 모델 추론 |
| Eigen3 | 행렬 연산 |
| yaml-cpp | 정책 설정 로딩 |
| zlib, cnpy | NPZ 파일 파싱 |

**bodyctrl_msgs 불필요** (모터 직접 제어 없음)

---

## 빌드 및 배포

### Main PC
```bash
ssh <user>@<MAIN_PC_IP>
cd ~/tklab_ws
colcon build --packages-select rl_control_new
source install/setup.bash
```

### Jetson Orin
```bash
ssh <user>@<orin_ip>
cd <workspace>
colcon build --packages-select hdmi_inference
source install/setup.bash
```

### 실행 순서
```bash
# Terminal 1 (Main PC): 바디 컨트롤
ros2 launch body_control body.launch.py

# Terminal 2 (Main PC): 모터 컨트롤 + 센서 발행
ros2 launch rl_control_new rl.launch.py

# Terminal 3 (Orin): 정책 추론
ros2 launch hdmi_inference hdmi_inference.launch.py
```

### 조작 순서
1. 시작 시 **STOP** 상태 (모터 현재 위치 유지)
2. **D키**(Yunzhuo) 또는 **X키**(Xbox) → **ZERO** 모드 (1.25초간 초기 자세로 이동)
3. **A+G키**(Yunzhuo) 또는 **A키**(Xbox) → **MLP** 모드 (정책 실행, 로봇 걷기 시작)
4. **C키**(Yunzhuo) 또는 **Y키**(Xbox) → **STOP** 모드 (즉시 정지)

---

## 참조 구현

C++ 구현은 Python sim2real 코드를 참조하여 작성됨:

| C++ 클래스 | Python 참조 | 역할 |
|-----------|------------|------|
| OnnxInference | `sim2real/rl_policy/utils/onnx_module.py` | ONNX 추론 |
| MotionData | `sim2real/rl_policy/utils/motion.py` | 모션 데이터 로드 |
| ObservationBuilder | `sim2real/rl_policy/observations/common.py` | 관측 빌더 |
| (command obs) | `sim2real/rl_policy/observations/motion.py` | 모션 레퍼런스 관측 |
| HdmiPolicyRunner | `sim2real/rl_policy/base_policy.py` | 정책 실행 통합 |

sim2real Python 코드 위치: `/home/kbc/sim2real/`
