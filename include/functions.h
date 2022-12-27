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

    vector<pair<pair<double, double>, vector<tuple<pair<double, double>, Eigen::Matrix3d, Eigen::Vector3d, vector<double>>>>>
    findSharedMatches(const vector<Eigen::Matrix3d> & R_is,
                      const vector<Eigen::Vector3d> & T_is,
                      const vector<vector<double>> & K_is,
                      const vector<vector<cv::Point2d>> & all_pts_q,
                      const vector<vector<cv::Point2d>> & all_pts_i);

    vector<string> getQueries(const string & queryList, const string & scene);

    tuple<string, Eigen::Matrix3d, Eigen::Vector3d, vector<double>,
            vector<string>,
            vector<Eigen::Matrix3d>, vector<Eigen::Vector3d>,
            vector<Eigen::Matrix3d>, vector<Eigen::Vector3d>,
            vector<vector<double>>,
            vector<vector<cv::Point2d>>, vector<vector<cv::Point2d>>>
          parseRelposeFile (const string & dir, const string & query, const string & fn);


    vector<int> optimizeSpacingZhou(const vector<Eigen::Vector3d> & old_centers,
                                    double l_thresh, double u_thresh, int N);

    void get_SG_points(const string & query, const string & db, vector<cv::Point2d> & pts_q, vector<cv::Point2d> & pts_i);

    void record_spaced(const string & query, const vector<string> & spaced, const string & folder);

    Eigen::Matrix3d smallRandomRotationMatrix(double s);

    pair<vector<cv::Point2d>, vector<cv::Point2d>> same_inliers(
            const vector<cv::Point2d> & pts_q, const vector<cv::Point2d> & pts_db,
            const Eigen::Matrix3d & R_qk, const Eigen::Vector3d & T_qk,
            const Eigen::Matrix3d & R_kq, const Eigen::Vector3d & T_kq,
            const Eigen::Matrix3d & K);

    double triangulateRays(const Eigen::Vector3d &ci, const Eigen::Vector3d &di, const Eigen::Vector3d &cj,
                           const Eigen::Vector3d &dj, Eigen::Vector3d &intersect);

    double getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2);

    double getPoint3DLineDist(const Eigen::Vector3d & h, const Eigen::Vector3d & c_k, const Eigen::Vector3d & v_k);

    double getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2);

    double rotationDifference(const Eigen::Matrix3d &r1, const Eigen::Matrix3d &r2);

    vector<double> getReprojectionErrors(const vector<cv::Point2d> & pts_to, const vector<cv::Point2d> & pts_from,
                                         const Eigen::Matrix3d & K, const Eigen::Matrix3d & R, const Eigen::Vector3d & T);

    pair<cv::Mat, vector<cv::KeyPoint>> getDescriptors(const string & image, const string & ext, const string & method);

    bool findMatches(double ratio,
                     const cv::Mat & desc_i, const vector<cv::KeyPoint> & kp_i,
                     const cv::Mat & desc_q, const vector<cv::KeyPoint> & kp_q,
                     vector<cv::Point2d> & pts_i, vector<cv::Point2d> & pts_q);

    bool findMatches(const string &db_image, const string &query_image, const string & ext, const string &method, double ratio,
                     vector<cv::Point2d> &pts_db, vector<cv::Point2d> &pts_query);

    bool findMatchesSorted(const string &db_image, const string & ext, const string &query_image, const string &method, double ratio,
                           vector<tuple<cv::Point2d, cv::Point2d, double>> & points);

    void findInliers(const Eigen::Matrix3d & R_q,
                     const Eigen::Vector3d & T_q,
                     const Eigen::Matrix3d & R_k,
                     const Eigen::Vector3d & T_k,
                     const Eigen::Matrix3d & K,
                     double threshold,
                     vector<cv::Point2d> & pts_q,
                     vector<cv::Point2d> & pts_db);

    int getRelativePose(const string & db_image, const string & ext, const string & query_image, const double * K, const string & method,
                        Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq);

    bool getRelativePose(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q,
                         const double * K,
                         double match_thresh,
                         Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq);

    int getRelativePose3D(const string &db_image, const string & ext, const string &query_image, const string &method,
                                    Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq);

    vector<int> optimizeSpacing(const Eigen::Vector3d & query,
                                const vector<Eigen::Vector3d> & centers,
                                int N, bool show_process);

    vector<int> randomSelection(int size, int N);

    void showKmeans(const vector<pair<Eigen::Vector3d, vector<pair<string, Eigen::Vector3d>>>> & clusters,
                    const vector<cv::Scalar> & colors);

    vector<string> kMeans(const vector<string> & images, int N, bool show_process);

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
    string getScene(const string & image, const string & mod);
    string getSequence(const string & image);
    vector<string> getTopN(const string& query_image, const string & ext, int N);
    void retrieveSimilar(const string & query_image, const string & replace, const string & ext, int max_num, double max_dist,
                         vector<string> & similar, vector<double> & distances);
    vector<string> spaceWithMostMatches(const string & query_image, const string & ext, const double * cam_matrix, int K,
                                        int N_thresh, double max_descriptor_dist, double separation, int min_matches,
                                        vector<Eigen::Matrix3d> & R_k,
                                        vector<Eigen::Vector3d> & t_k,
                                        vector<Eigen::Matrix3d> & R_qk,
                                        vector<Eigen::Vector3d> & t_qk);
    map<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<double>, vector<double>, vector<pair<cv::Point2d, cv::Point2d>>>>
    getRelativePoses(const string & query, const string & data_folder);

//// Visualization

    void showTop(int rows, int cols,
                 const string & query_image, const vector<string> & returned,
                 const string & ext, const string & title);

    void showSpaced(const string & query_image, const vector<string> & spaced, const string & ext);

    void showSpacedInTop(int rows, int cols,
                         const string & query_image, const vector<string> & returned,
                         const vector<string> & spaced, const string & ext);

    void showTop1000(const string & query_image, const string & ext, int max_num, double max_descriptor_dist, int inliers);

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