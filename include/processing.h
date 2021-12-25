//
// Created by Cameron Fiore on 12/15/21.
//

#ifndef IMAGEMATCHERPROJECT_PROCESSING_H
#define IMAGEMATCHERPROJECT_PROCESSING_H

#endif //IMAGEMATCHERPROJECT_PROCESSING_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <base/database.h>
#include <base/image.h>

using namespace std;

namespace processing {
    double triangulateRays(const Eigen::Vector3d &ci, const Eigen::Vector3d &di, const Eigen::Vector3d &cj,
                           const Eigen::Vector3d &dj, Eigen::Vector3d &intersect);

    double getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

    double getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2);

    double rotationDifference(const Eigen::Matrix3d &r1, const Eigen::Matrix3d &r2);

    bool
    findMatches(const string &db_image, const string &query_image, const string &method, vector<cv::Point2d> &pts_db,
                vector<cv::Point2d> &pts_query);

    void getRelativePose(const string &db_image, const string &query_image, const string &method, Eigen::Matrix3d &R_kq,
                         Eigen::Vector3d &t_kq);

    Eigen::Vector3d
    hypothesizeQueryCenter(const string &query_image, const vector<string> &ensemble, const string &dataset);

    Eigen::Vector3d
    hypothesizeQueryCenter(const vector<Eigen::Matrix3d> &R_k,
                           const vector<Eigen::Vector3d> &t_k,
                           const vector<Eigen::Matrix3d> &R_qk,
                           const vector<Eigen::Vector3d> &t_qk);

    template<typename DataType, typename ForwardIterator>
    Eigen::Quaternion<DataType> averageQuaternions(ForwardIterator const &begin, ForwardIterator const &end);

    Eigen::Matrix3d rotationAverage(const vector<Eigen::Matrix3d> &rotations);

    void getEnsemble(const int & max_size,
                     const string & dataset,
                     const string & query,
                     const vector<string> & images,
                     vector<Eigen::Matrix3d> & R_k,
                     vector<Eigen::Vector3d> & t_k,
                     vector<Eigen::Matrix3d> & R_qk,
                     vector<Eigen::Vector3d> & t_qk);

    void useRANSAC(const vector<Eigen::Matrix3d> &R_k,
                              const vector<Eigen::Vector3d> &t_k,
                              const vector<Eigen::Matrix3d> &R_qk,
                              const vector<Eigen::Vector3d> &t_qk,
                              Eigen::Matrix3d & R_q,
                              Eigen::Vector3d & c_q);
}