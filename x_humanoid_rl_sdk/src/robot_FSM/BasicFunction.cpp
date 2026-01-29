#include "BasicFunction.h"

Eigen::Vector4d rpy_to_quat(const double r, const double p, const double y) {
  const double roll  = r;
  const double pitch = p;
  const double yaw   = y;

  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);

  const double w = cr * cp * cy + sr * sp * sy;
  const double x = sr * cp * cy - cr * sp * sy;
  const double ya = cr * sp * cy + sr * cp * sy;
  const double z = cr * cp * sy - sr * sp * cy;
  return Eigen::Vector4d(w,x,ya,z);
}

Eigen::Matrix3d matrix_from_quat(const Eigen::Vector4d& q_wxyz) {
  const double r = q_wxyz(0); // w
  const double i = q_wxyz(1); // x
  const double j = q_wxyz(2); // y
  const double k = q_wxyz(3); // z

  const double n2 = r*r + i*i + j*j + k*k;
  if (n2 <= 1e-24) {
    return Eigen::Matrix3d::Identity(); // 너무 작으면 안전하게 identity
  }

  const double two_s = 2.0 / n2;

  Eigen::Matrix3d R;
  R(0,0) = 1.0 - two_s * (j*j + k*k);
  R(0,1) =        two_s * (i*j - k*r);
  R(0,2) =        two_s * (i*k + j*r);

  R(1,0) =        two_s * (i*j + k*r);
  R(1,1) = 1.0 - two_s * (i*i + k*k);
  R(1,2) =        two_s * (j*k - i*r);

  R(2,0) =        two_s * (i*k - j*r);
  R(2,1) =        two_s * (j*k + i*r);
  R(2,2) = 1.0 - two_s * (i*i + j*j);

  return R;
}


Eigen::Vector4d subtract_frame_transforms(const Eigen::Vector4d& q01,const Eigen::Vector4d& b){
  Eigen::Quaterniond qa(q01(0), -q01(1), -q01(2), -q01(3));
  Eigen::Quaterniond qb(b(0), b(1), b(2), b(3));
  Eigen::Quaterniond qc = qa * qb;               // Hamilton product
  return Eigen::Vector4d(qc.w(), qc.x(), qc.y(), qc.z());
}
Eigen::Vector4d quat_mul(const Eigen::Vector4d& a,
                                         const Eigen::Vector4d& b) {
  Eigen::Quaterniond qa(a(0), a(1), a(2), a(3));  // (w,x,y,z)
  Eigen::Quaterniond qb(b(0), b(1), b(2), b(3));
  Eigen::Quaterniond qc = qa * qb;               // Hamilton product
  return Eigen::Vector4d(qc.w(), qc.x(), qc.y(), qc.z());
}

Eigen::Vector4d remove_yaw_offset(const Eigen::Vector4d& quat_wxyz, const double yaw_offset){
  Eigen::Vector4d yaw_quat = rpy_to_quat(0.0, 0.0, -yaw_offset);
  Eigen::Vector4d result = quat_mul(yaw_quat, quat_wxyz);
  return result;
}

double quat_yaw(const Eigen::Vector4d& q_in) {
  Eigen::Quaterniond q(q_in(0),q_in(1),q_in(2),q_in(3));
  const double w = q.w();
  const double x = q.x();
  const double y = q.y();
  const double z = q.z();

  // roll (x-axis)
  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis)
  double sinp = 2.0 * (w * y - z * x);
  sinp = std::clamp(sinp, -1.0, 1.0);
  const double pitch = std::asin(sinp);

  // yaw (z-axis)
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  const double yaw = std::atan2(siny_cosp, cosy_cosp);

  // return Eigen::Vector3d(roll, pitch, yaw);
  return yaw;
}

Eigen::Matrix3d RotX(double x) {
  return Eigen::AngleAxisd(x, Eigen::Vector3d::UnitX()).toRotationMatrix();
}
Eigen::Matrix3d RotY(double y) {
  return Eigen::AngleAxisd(y, Eigen::Vector3d::UnitY()).toRotationMatrix();
}
Eigen::Matrix3d RotZ(double z) {
  return Eigen::AngleAxisd(z, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

void Euler_XYZToMatrix(Eigen::Matrix3d &R, Eigen::Vector3d euler_a) {
  R = Eigen::AngleAxisd(euler_a[0], Eigen::Vector3d::UnitX()).toRotationMatrix()
      * Eigen::AngleAxisd(euler_a[1], Eigen::Vector3d::UnitY()).toRotationMatrix()
      * Eigen::AngleAxisd(euler_a[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
}
void EulerZYXToMatrix(Eigen::Matrix3d &R, Eigen::Vector3d euler_a) {
  R = Eigen::AngleAxisd(euler_a[0], Eigen::Vector3d::UnitZ()).toRotationMatrix()
      * Eigen::AngleAxisd(euler_a[1], Eigen::Vector3d::UnitY()).toRotationMatrix()
      * Eigen::AngleAxisd(euler_a[2], Eigen::Vector3d::UnitX()).toRotationMatrix();
}
void MatrixToEulerXYZ(Eigen::Matrix3d R, Eigen::Vector3d &euler) {
  euler.setZero();
  // y (-pi/2 pi/2)
  euler(1) = asin(R(0, 2));
  // z [-pi pi]
  double sinz = -R(0, 1) / cos(euler(1));
  double cosz = R(0, 0) / cos(euler(1));
  euler(2) = atan2(sinz, cosz);
  // x [-pi pi]
  double sinx = -R(1, 2) / cos(euler(1));
  double cosx = R(2, 2) / cos(euler(1));
  euler(0) = atan2(sinx, cosx);
}

void clip(Eigen::VectorXd &in_, double lb, double ub) {
  for (int i = 0; i < in_.size(); i++) {
    if (in_[i] < lb) { in_[i] = lb; }
    if (in_[i] > ub) { in_[i] = ub; }
  }
}

double clip(double a, double lb, double ub) {
  double b = a;
  if (a < lb) { b = lb; }
  if (a > ub) { b = ub; }
  return b;
}

Eigen::VectorXd gait_phase(double timer,
                           double gait_cycle_,
                           double left_theta_offset_,
                           double right_theta_offset_,
                           double left_phase_ratio_,
                           double right_phase_ratio_) {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(6);
  double left_phase = (timer / gait_cycle_ + left_theta_offset_) - floor(timer / gait_cycle_ + left_theta_offset_);
  double right_phase = (timer / gait_cycle_ + right_theta_offset_) - floor(timer / gait_cycle_ + right_theta_offset_);
  res(0) = sin(2.0 * M_PI * left_phase);
  res(1) = sin(2.0 * M_PI * right_phase);
  res(2) = cos(2.0 * M_PI * left_phase);
  res(3) = cos(2.0 * M_PI * right_phase);
  res(4) = left_phase_ratio_;
  res(5) = right_phase_ratio_;


  return res;
}


void FifthPoly(Eigen::VectorXd p0, Eigen::VectorXd p0_dot, Eigen::VectorXd p0_dotdot,
               Eigen::VectorXd p1, Eigen::VectorXd p1_dot, Eigen::VectorXd p1_dotdot,
               double totalTime,       // total time
               double currentTime,     //current time,from 0 to total time
               Eigen::VectorXd &pd, Eigen::VectorXd &pd_dot, Eigen::VectorXd &pd_dotdot) {
  double t = currentTime;
  double time = totalTime;
  if (t < totalTime) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 6);
    A << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 0.0,
        -10 / pow(time, 3), -6 / pow(time, 2), -3 / (2 * time), 10 / pow(time, 3), -4 / pow(time, 2), 1 / (2 * time),
        15 / pow(time, 4), 8 / pow(time, 3), 3 / (2 * pow(time, 2)), -15 / pow(time, 4), 7 / pow(time, 3), -1 / pow(time, 2),
        -6 / pow(time, 5), -3 / pow(time, 4), -1 / (2 * pow(time, 3)), 6 / pow(time, 5), -3 / pow(time, 4), 1 / (2 * pow(time, 3));
    Eigen::MatrixXd x0 = Eigen::MatrixXd::Zero(6, 1);
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(6, 1);
    for (int i = 0; i < p0.size(); i++) {
      x0 << p0(i), p0_dot(i), p0_dotdot(i), p1(i), p1_dot(i), p1_dotdot(i);
      a = A * x0;
      pd(i) = a(0) + a(1) * t + a(2) * t * t + a(3) * t * t * t + a(4) * t * t * t * t + a(5) * t * t * t * t * t;
      pd_dot(i) = a(1) + 2 * a(2) * t + 3 * a(3) * t * t + 4 * a(4) * t * t * t + 5 * a(5) * t * t * t * t;
      pd_dotdot(i) = 2 * a(2) + 6 * a(3) * t + 12 * a(4) * t * t + 20 * a(5) * t * t * t;
    }
  } else {
    pd = p1;
    pd_dot = p1_dot;
    pd_dotdot = p1_dotdot;
  }
}

LowPassFilter :: LowPassFilter ( double cutOffFreq, double dampRatio, 
                                 double dTime , int nFilter ) {
    dT = dTime;
	sigIn_1 = Eigen::VectorXd::Zero(nFilter);
    sigIn_2 = Eigen::VectorXd::Zero(nFilter);
    sigOut_1 = Eigen::VectorXd::Zero(nFilter);
    sigOut_2 = Eigen::VectorXd::Zero(nFilter);

	double freqInRad = 2. * M_PI * cutOffFreq;
	double c = 2.0 / dT;
	double sqrC = c * c;
	double sqrW = freqInRad * freqInRad;

	b2 = sqrC + 2.0 * dampRatio * freqInRad * c + sqrW;
	b1 = -2.0 * ( sqrC - sqrW );
	b0 = sqrC - 2.0 * dampRatio * freqInRad * c + sqrW;

    a2 = sqrW;
    a1 = 2.0 * sqrW;
    a0 = sqrW;
    a2 /= b2;
    a1 /= b2;
    a0 /= b2;

    b1 /= b2;
    b0 /= b2;
    b2 = 1.0;
}

Eigen::VectorXd LowPassFilter::mFilter ( Eigen::VectorXd sigIn ) {

	Eigen::VectorXd sigOut = a2 * sigIn + a1 * sigIn_1 + 
               a0 * sigIn_2 - b1 * sigOut_1 - b0 * sigOut_2;
    sigIn_2 = sigIn_1;
	sigIn_1 = sigIn;
	sigOut_2 = sigOut_1;
	sigOut_1 = sigOut;
    return sigOut;
}
