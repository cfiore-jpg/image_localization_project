//
// Created by Cameron Fiore on 1/23/22.
//

#ifndef IMAGE_LOCALIZATION_PROJECT_POSEESTIMATION_H
#define IMAGE_LOCALIZATION_PROJECT_POSEESTIMATION_H

#endif //IMAGE_LOCALIZATION_PROJECT_POSEESTIMATION_H

#include "../include/imageMatcher_Orb.h"
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/Space.h"
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

namespace pose {

    Eigen::Vector3d hypothesizeQueryCenter (const vector<Eigen::Matrix3d> &R_k,
                                           const vector<Eigen::Vector3d> &t_k,
                                           const vector<Eigen::Matrix3d> &R_qk,
                                           const vector<Eigen::Vector3d> &t_qk);

    Eigen::Vector3d hypothesizeQueryCenterRANSAC (const double & inlier_thresh,
                                                  vector<Eigen::Matrix3d> & R_k,
                                                  vector<Eigen::Vector3d> & t_k,
                                                  vector<Eigen::Matrix3d> & R_qk,
                                                  vector<Eigen::Vector3d> & t_qk,
                                                  vector<vector<tuple<cv::Point2d, cv::Point2d, double>>> & all_points);

    Eigen::Matrix3d hypothesizeQueryRotation (const Eigen::Vector3d & c_q,
                                              const Eigen::Matrix3d & R_q,
                                              const vector<Eigen::Matrix3d> & R_k,
                                              const vector<Eigen::Vector3d> & t_k,
                                              const vector<Eigen::Matrix3d> & R_qk,
                                              const vector<Eigen::Vector3d> &t_qk);

    tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, int, double, double> hypothesizeRANSAC (
            const double & t_thresh,
            const double & r_thresh,
            const vector<int> & mask,
            const vector<Eigen::Matrix3d> & R_k,
            const vector<Eigen::Vector3d> & t_k,
            const vector<Eigen::Matrix3d> & R_qk,
            const vector<Eigen::Vector3d> & t_qk,
            const Eigen::Vector3d & c_q,
            const Eigen::Matrix3d & R_q);

    void adjustHypothesis (const vector<Eigen::Matrix3d> & R_k,
                           const vector<Eigen::Vector3d> & T_k,
                           const vector<vector<tuple<cv::Point2d, cv::Point2d, double>>> & all_points,
                           const double & error_thresh,
                           const double * K,
                           Eigen::Matrix3d & R_q,
                           Eigen::Vector3d & T_q);

    template<typename DataType, typename ForwardIterator>
    Eigen::Quaternion<DataType> averageQuaternions(ForwardIterator const &begin, ForwardIterator const &end);

    Eigen::Matrix3d rotationAverage(const vector<Eigen::Matrix3d> &rotations);
}