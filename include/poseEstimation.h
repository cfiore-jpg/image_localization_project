//
// Created by Cameron Fiore on 1/23/22.
//

#ifndef IMAGE_LOCALIZATION_PROJECT_POSEESTIMATION_H
#define IMAGE_LOCALIZATION_PROJECT_POSEESTIMATION_H

#endif //IMAGE_LOCALIZATION_PROJECT_POSEESTIMATION_H

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

    Eigen::Vector3d estimate3Dpoint(const vector<tuple<pair<double, double>, Eigen::Matrix3d, Eigen::Vector3d, vector<double>>> & matches);

    pair<Eigen::Vector3d, vector<tuple<pair<double, double>, Eigen::Matrix3d, Eigen::Vector3d, vector<double>>>>
    RANSAC3DPoint(double inlier_thresh, const vector<tuple<pair<double, double>, Eigen::Matrix3d, Eigen::Vector3d, vector<double>>> & matches);

    cv::Point2d reproject3Dto2D(const Eigen::Vector3d & point3d,
                                const Eigen::Matrix3d & R,
                                const Eigen::Vector3d & T,
                                const vector<double> & K);

    cv::Point2d undistort_point(const cv::Point2d & pt, double f, double cx, double cy, double rn);

    double reprojError(const Eigen::Vector3d & point3d,
                             const Eigen::Matrix3d & R,
                             const Eigen::Vector3d & T,
                             const vector<double> & K,
                             double mx, double my);

    void estimatePose(const vector<Eigen::Matrix3d> & R_ks,
                      const vector<Eigen::Vector3d> & T_ks,
                      const vector<Eigen::Vector3d> & T_qks,
                      Eigen::Matrix3d & R_q,
                      Eigen::Vector3d & c_q);

    /// BUNDLE ADJUSTMENT
    void sceneBundleAdjust(int num_ims, const double K[4],
                           const string &folder, const string &scene, const string &sequence, const string &frame,
                           const string &pose_ext, const string &rgb_ext);


    /// PERFORMANCE TESTING
    double averageReprojectionError(const double K[4], const string &query, const Eigen::Matrix3d &R_q,
                                    const Eigen::Vector3d &T_q,
                                    const vector<string> &anchors, const vector<Eigen::Matrix3d> &R_ks,
                                    const vector<Eigen::Vector3d> &T_ks);

    double tripletCircles(const double K[4], const vector<pair<cv::Mat, vector<cv::KeyPoint>>> & desc_kp_anchors);


    /// CENTER HYPOTHESIS
    Eigen::Vector3d c_q_closed_form(const vector<Eigen::Matrix3d> &R_ks,
                                    const vector<Eigen::Vector3d> &T_ks,
                                    const vector<Eigen::Matrix3d> &R_qks,
                                    const vector<Eigen::Vector3d> &T_qks);


    //// TRANSLATION HYPOTHESIS
    Eigen::Vector3d T_q_govindu(const vector<Eigen::Vector3d> &T_ks,
                                const vector<Eigen::Matrix3d> &R_qks,
                                const vector<Eigen::Vector3d> &T_qks);

    pair<Eigen::Vector3d, Eigen::Vector3d> T_q_closed_form (const vector<Eigen::Matrix3d> & R_ks,
                                                            const vector<Eigen::Vector3d> & T_ks,
                                                            const vector<Eigen::Matrix3d> & R_qks,
                                                            const vector<Eigen::Vector3d> & T_qks);


    //// ROTATION HYPOTHESIS
    template<typename DataType, typename ForwardIterator>
    Eigen::Quaternion<DataType> averageQuaternions(ForwardIterator const &begin, ForwardIterator const &end);

    Eigen::Matrix3d R_q_average(const vector<Eigen::Matrix3d> &rotations);

    Eigen::Matrix3d R_q_closed_form(bool use_Rqk, bool normalize, bool version,
                                    const Eigen::Vector3d &c_q,
                                    const vector<Eigen::Matrix3d> &R_ks,
                                    const vector<Eigen::Vector3d> &T_ks,
                                    const vector<Eigen::Matrix3d> &R_qks,
                                    const vector<Eigen::Vector3d> &T_qks);


    //// RANSAC
    tuple<Eigen::Vector3d,
//            Eigen::Vector3d,
//            Eigen::Vector3d,
            Eigen::Matrix3d,
//            Eigen::Matrix3d,
//            Eigen::Matrix3d,
//            Eigen::Matrix3d,
//            Eigen::Matrix3d,
//            Eigen::Matrix3d,
//            Eigen::Matrix3d,
            vector<int>>
            hypothesizeRANSAC(double threshold,
                      const vector<Eigen::Matrix3d> & R_ks,
                      const vector<Eigen::Vector3d> & T_ks,
                      const vector<Eigen::Matrix3d> & R_qks,
                      const vector<Eigen::Vector3d> & T_qks);


    //// FINAL POSE ADJUSTMENT
    pair<vector<cv::Point2d>, vector<Eigen::Vector3d>>
    adjustHypothesis (const vector<Eigen::Matrix3d> & R_is,
                      const vector<Eigen::Vector3d> & T_is,
                      const vector<vector<double>> & K_is,
                      const vector<double> & K_q,
                      const vector<vector<cv::Point2d>> & all_pts_q,
                      const vector<vector<cv::Point2d>> & all_pts_i,
                      double covis,
                      double pixel_thresh,
                      double post_ransac,
                      double reproj_tolerance,
                      double cauchy,
                      Eigen::Matrix3d & R_q,
                      Eigen::Vector3d & T_q);


    void visualizeRelpose(const string & query,
                          const vector<string> & anchors,
                          const vector<Eigen::Matrix3d> & R_is,
                          const vector<Eigen::Vector3d> & T_is,
                          const vector<Eigen::Matrix3d> & R_qis,
                          const vector<Eigen::Vector3d> & T_qis,
                          const vector<vector<double>> & K_is,
                          const vector<double> & K_q,
                          const vector<vector<cv::Point2d>> & all_pts_q,
                          const vector<vector<cv::Point2d>> & all_pts_i,
                          const Eigen::Matrix3d & R_q_before,
                          const Eigen::Vector3d & T_q_before,
                          const Eigen::Matrix3d & R_q_after,
                          const Eigen::Vector3d & T_q_after);
}