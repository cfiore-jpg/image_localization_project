//
// Created by Cameron Fiore on 12/15/21.
//

#ifndef IMAGEMATCHERPROJECT_UTILS_H
#define IMAGEMATCHERPROJECT_UTILS_H

#endif //IMAGEMATCHERPROJECT_UTILS_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

namespace functions {

    Eigen::Matrix3d smallRandomRotationMatrix();

    double triangulateRays(const Eigen::Vector3d &ci, const Eigen::Vector3d &di, const Eigen::Vector3d &cj,
                           const Eigen::Vector3d &dj, Eigen::Vector3d &intersect);

    double getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

    double getPoint3DLineDist(const Eigen::Vector3d & h, const Eigen::Vector3d & c_k, const Eigen::Vector3d & v_k);

    double getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2);

    double rotationDifference(const Eigen::Matrix3d &r1, const Eigen::Matrix3d &r2);

    bool findMatches(const string &db_image, const string &query_image, const string &method, double ratio,
                     vector<cv::Point2d> &pts_db, vector<cv::Point2d> &pts_query);

    bool findMatchesSorted(const string &db_image, const string &query_image, const string &method, double ratio,
                           vector<tuple<cv::Point2d, cv::Point2d, double>> & points);

    vector<pair<cv::Point2d, cv::Point2d>> findInliersForFundamental(const Eigen::Matrix3d & F, double threshold,
                                                                     const vector<tuple<cv::Point2d, cv::Point2d, double>> & points);

    int getRelativePose(const string & db_image, const string & query_image, const double * K, const string & method,
                        Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq);

    bool getRelativePose(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, const double * K,
                        Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq);

    int getRelativePose3D(const string &db_image, const string &query_image, const string &method,
                                    Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq);

    vector<string> optimizeSpacing(const vector<string> & images, int N, bool show_process, const string & dataset);

    void createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene);
    void createQueryVector(vector<string> &listQuery, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene);

////SURF Functions
//void loadFeaturesSURF(const vector<string> &listImage, vector<vector<vector<float>>> &features);
//void changeStructureSURF(const vector<float> &plain, vector<vector<float>> &out, int L);
//void DatabaseSaveSURF(const vector<vector<vector<float>>> &features);

//// ORB functions
    void loadFeaturesORB(const vector<string> &listImage, vector<vector<cv::Mat>> &features);
    void changeStructureORB(const cv::Mat &plain, vector<cv::Mat> &out);
    void DatabaseSaveORB(const vector<vector<cv::Mat>> &features);

//// Image Similarity Functions
    void saveResultVector(const vector<pair<int, float>>& result, const string& query_image, const string& method, const string & num);
    vector<pair<int, float>> getResultVector(const string& query_image, const string& method, const string & num);
    string getScene(const string & image);
    string getSequence(const string & image);
    vector<string> getTopN(const string& query_image, int N);
    vector<string> retrieveSimilar(const string& query_image, int max_num, double max_descriptor_dist);
    vector<string> spaceWithMostMatches(const string & query_image, const double * cam_matrix, int K,
                                        int N_thresh, double max_descriptor_dist, double separation, int min_matches,
                                        vector<Eigen::Matrix3d> & R_k,
                                        vector<Eigen::Vector3d> & t_k,
                                        vector<Eigen::Matrix3d> & R_qk,
                                        vector<Eigen::Vector3d> & t_qk);

//// Visualization
    void showTop1000(const string & query_image, int max_num, double max_descriptor_dist, int inliers);
    cv::Mat projectCentersTo2D(const string & query, const vector<string> & images,
                               unordered_map<string, cv::Scalar> & seq_colors,
                               const string & Title);
    double drawLines(cv::Mat & im1, cv::Mat & im2,
                     const vector<cv::Point3d> & lines_draw_on_1,
                     const vector<cv::Point3d> & lines_draw_on_2,
                     const vector<cv::Point2d> & pts1,
                     const vector<cv::Point2d> & pts2,
                     const vector<cv::Scalar> & colors);
    cv::Mat showAllImages(const string & query, const vector<string> & images,
                          unordered_map<string, cv::Scalar> & seq_colors,
                          const unordered_set<string> & unincluded_all,
                          const unordered_set<string> & unincluded_top1000,
                          const string & Title);

}