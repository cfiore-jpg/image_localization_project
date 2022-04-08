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

    Eigen::Matrix3d solve_rotation(const Eigen::Vector3d & c_q,
                                   const vector<Eigen::Matrix3d> & R_k,
                                   const vector<Eigen::Vector3d> & t_k,
                                   const vector<Eigen::Matrix3d> & R_qk,
                                   const vector<Eigen::Vector3d> & t_qk);

    Eigen::Matrix3d solve_rotation_with_norm(const Eigen::Vector3d & c_q,
                                                       const vector<Eigen::Matrix3d> & R_k,
                                                       const vector<Eigen::Vector3d> & t_k,
                                                       const vector<Eigen::Matrix3d> & R_qk,
                                                       const vector<Eigen::Vector3d> & t_qk);

    Eigen::MatrixXcd solver_problem_averageRQuatMetric_red(const Eigen::VectorXd& data);

}