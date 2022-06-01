//
// Created by Cameron Fiore on 3/9/22.
//

#ifndef IMAGE_LOCALIZATION_PROJECT_OPTIMALROTATIONSOLVER_H
#define IMAGE_LOCALIZATION_PROJECT_OPTIMALROTATIONSOLVER_H

#endif //IMAGE_LOCALIZATION_PROJECT_OPTIMALROTATIONSOLVER_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;

namespace rotation {

    Eigen::Matrix3d optimal_rotation_using_R_qks(const Eigen::Vector3d &c_q,
                                                 const vector<Eigen::Matrix3d> &R_ks,
                                                 const vector<Eigen::Vector3d> &T_ks,
                                                 const vector<Eigen::Matrix3d> &R_qks,
                                                 const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d optimal_rotation_using_T_qks(const Eigen::Vector3d &c_q,
                                                 const vector<Eigen::Matrix3d> &R_ks,
                                                 const vector<Eigen::Vector3d> &T_ks,
                                                 const vector<Eigen::Matrix3d> &R_qks,
                                                 const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d NORM_optimal_rotation_using_R_qks(const Eigen::Vector3d &c_q,
                                                      const vector<Eigen::Matrix3d> &R_ks,
                                                      const vector<Eigen::Vector3d> &T_ks,
                                                      const vector<Eigen::Matrix3d> &R_qks,
                                                      const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d NORM_optimal_rotation_using_T_qks(const Eigen::Vector3d &c_q,
                                                      const vector<Eigen::Matrix3d> &R_ks,
                                                      const vector<Eigen::Vector3d> &T_ks,
                                                      const vector<Eigen::Matrix3d> &R_qks,
                                                      const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d optimal_rotation_using_T_qks_NEW(const Eigen::Vector3d &c_q,
                                                          const vector<Eigen::Matrix3d> &R_ks,
                                                          const vector<Eigen::Vector3d> &T_ks,
                                                          const vector<Eigen::Matrix3d> &R_qks,
                                                          const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d optimal_rotation_using_R_qks_NEW(const Eigen::Vector3d &c_q,
                                                          const vector<Eigen::Matrix3d> &R_ks,
                                                          const vector<Eigen::Vector3d> &T_ks,
                                                          const vector<Eigen::Matrix3d> &R_qks,
                                                          const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d optimal_rotation_using_R_qks_NEW_2(const Eigen::Vector3d &c_q,
                                                       const vector<Eigen::Matrix3d> &R_ks,
                                                       const vector<Eigen::Vector3d> &T_ks,
                                                       const vector<Eigen::Matrix3d> &R_qks,
                                                       const vector<Eigen::Vector3d> &T_qks);

    Eigen::Matrix3d optimal_rotation_using_T_qks_NEW_2(const Eigen::Vector3d &c_q,
                                                       const vector<Eigen::Matrix3d> &R_ks,
                                                       const vector<Eigen::Vector3d> &T_ks,
                                                       const vector<Eigen::Matrix3d> &R_qks,
                                                       const vector<Eigen::Vector3d> &T_qks);

    Eigen::MatrixXcd solver_problem_averageRQuatMetric_red(const Eigen::VectorXd &data);

    Eigen::MatrixXcd solver_problem_averageRQuatMetricAlter_red_new(const Eigen::VectorXd &data);

}