//
// Created by Cameron Fiore on 6/30/21.
//

#ifndef IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H
#define IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H

#endif //IMAGEMATCHERPROJECT_ORB_SEVENSCENES_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <base/database.h>
#include <base/image.h>

using namespace std;

namespace sevenScenes
{

    Eigen::Matrix3d getR(const string& image);

    Eigen::Vector3d getT(const string& image);

    double getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

    double getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2);

    vector<tuple<string, string, vector<string>, vector<string>>> createInfoVector();

    vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> pairSelection(const vector<pair<int, float>> &result, const vector<string> &listImage, const string &query, double loThresh, int maxNum, int method);

    vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> findXImagesWithYPoints(const vector<pair<int, float>> &result, const vector<string> &listImage, const string &query, int method, int X, int Y);

    vector<Eigen::Vector3d> findAllHypotheses(const vector<tuple<Eigen::Vector3d, Eigen::Vector3d, int>> &processed);

    bool findEssentialMatrix(const string &db_image, const string &query_image, int method, Eigen::Matrix3d &R, Eigen::Vector3d &t, int& num_points);

    void putInCorrectCluster(const cv::Point2d &q_point, const cv::Point2d &db_point, vector<pair<cv::Point2d, vector<pair<cv::Point2d, cv::Point2d>>>> &clusters);

    bool converged(const vector<pair<cv::Point2d, vector<pair<cv::Point2d, cv::Point2d>>>> &old_clusters, const vector<pair<cv::Point2d, vector<pair<cv::Point2d, cv::Point2d>>>> &new_clusters);

    vector<pair<vector<cv::Point2d>, vector<cv::Point2d>>> findClusters(const vector<cv::Point2d> &q_points, const vector<cv::Point2d> &db_points, int num_clusters, cv::Mat im);

    double findFarthest(const vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d>> &rel_images, const Eigen::Vector3d &avg);

    Eigen::Vector3d findStandardDev(const vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d>> &rel_images, const Eigen::Vector3d &avg);

    bool withinXDevs(const Eigen::Vector3d &dev, const Eigen::Vector3d &avg, const Eigen::Vector3d &vec, int numDevs);

    double triangulateRays(const Eigen::Vector3d& ci, const Eigen::Vector3d& di, const Eigen::Vector3d& cj, const Eigen::Vector3d& dj, Eigen::Vector3d &intersect);

    void findCorrectRiqs(const cv::Mat& E1, const Eigen::Matrix3d& R1, const cv::Mat& E2, const Eigen::Matrix3d& R2, Eigen::Matrix3d &Riq1, Eigen::Matrix3d &Riq2);

    Eigen::Vector3d useRANSAC(const vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> &rel_images, const double &inlier_thresh);

    Eigen::Vector3d useRANSAC_v2(const vector<Eigen::Vector3d> &hypotheses, double inlier_thresh);
}




