

#ifndef BASICFUNCTION_H_
#define BASICFUNCTION_H_

#include <Eigen/Dense>
double quat_yaw(Eigen::Vector4d q_in);
Eigen::Vector4d rpy_to_quat(const double r, const double p, const double y);
Eigen::Vector4d quat_mul(const Eigen::Vector4d& a, const Eigen::Vector4d& b);
Eigen::Vector4d subtract_frame_transforms(const Eigen::Vector4d& q01,const Eigen::Vector4d& b);
Eigen::Matrix3d matrix_from_quat(const Eigen::Vector4d& q_wxyz);
Eigen::Vector4d remove_yaw_offset(const Eigen::Vector4d& quat_wxyz, const double yaw_offset);

Eigen::Matrix3d RotX(double x);
Eigen::Matrix3d RotY(double y);
Eigen::Matrix3d RotZ(double z);
void Euler_XYZToMatrix(Eigen::Matrix3d &R, Eigen::Vector3d euler_a);
void EulerZYXToMatrix(Eigen::Matrix3d &R, Eigen::Vector3d euler_a);
void MatrixToEulerXYZ(Eigen::Matrix3d R, Eigen::Vector3d &euler);
void clip(Eigen::VectorXd &in_, double lb, double ub);
double clip(double a, double lb, double ub);
Eigen::VectorXd gait_phase(double timer,
                           double gait_cycle_,
                           double left_theta_offset_,
                           double right_theta_offset_,
                           double left_phase_ratio_,
                           double right_phase_ratio_);

void FifthPoly(Eigen::VectorXd p0, Eigen::VectorXd p0_dot, Eigen::VectorXd p0_dotdot,     // start point states
               Eigen::VectorXd p1, Eigen::VectorXd p1_dot, Eigen::VectorXd p1_dotdot,     // end point states
               double totalTime,       // total time
               double currentTime,     //current time,from 0 to total time
               Eigen::VectorXd &pd, Eigen::VectorXd &pd_dot, Eigen::VectorXd &pd_dotdot);// output command

class LowPassFilter {
	public:
		LowPassFilter ( double cutOffFreq, double dampRatio, double dTime, int nFilter );
		Eigen::VectorXd mFilter ( Eigen::VectorXd sigIn );
		
	private:
		double dT;
        Eigen::VectorXd sigIn_1;
        Eigen::VectorXd sigIn_2;
        Eigen::VectorXd sigOut_1;
        Eigen::VectorXd sigOut_2;
		double b2, b1, b0;
		double a2, a1, a0;
};

#endif //BASICFUNCTION_H_
