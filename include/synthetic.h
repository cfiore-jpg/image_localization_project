//
// Created by Cameron Fiore on 2/15/22.
//

#ifndef IMAGE_LOCALIZATION_PROJECT_SYNTHETIC_H
#define IMAGE_LOCALIZATION_PROJECT_SYNTHETIC_H

#endif //IMAGE_LOCALIZATION_PROJECT_SYNTHETIC_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

namespace synthetic {

    vector<string> getAll();

    vector<Eigen::Vector3d> get3DPoints();

    vector<string> omitQuery(int i, const vector<string> & all_ims);

    Eigen::Matrix3d getR(const string& image);

    Eigen::Vector3d getC(const string& image);

    void getAbsolutePose(const string& image, Eigen::Matrix3d &R_i, Eigen::Vector3d &t_i);

    void findSyntheticMatches(const string & db_im, const string & query, vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q);

    void addGaussianNoise(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, double std_dev);

    void addOcclusion(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, int n);

    void addMismatches(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, double percentage);

}