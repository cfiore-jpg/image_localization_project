#include "include/aachen.h"
#include "include/sevenScenes.h"
#include "include/synthetic.h"
#include "include/functions.h"
#include "include/poseEstimation.h"
#include <iostream>
#include <fstream>
#include <Eigen/SVD>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "include/OptimalRotationSolver.h"
#include "include/CambridgeLandmarks.h"


#define PI 3.1415926536

using namespace std;
using namespace chrono;

mutex mtx;

void findInliers (double threshold,
                  int i,
                  int j,
                  const vector<Eigen::Matrix3d> * R_ks,
                  const vector<Eigen::Vector3d> * T_ks,
                  const vector<Eigen::Matrix3d> * R_qks,
                  const vector<Eigen::Vector3d> * T_qks,
                  vector<tuple<int, int, double, vector<int>>> * results) {

    int K = int((*R_ks).size());

    vector<Eigen::Matrix3d> set_R_ks {(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks {(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks {(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks {(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};
    double score = 0;

    Eigen::Matrix3d R_h = pose::R_q_average(vector<Eigen::Matrix3d>{(*R_qks)[i] * (*R_ks)[i], (*R_qks)[j] * (*R_ks)[j]});
    Eigen::Vector3d c_h = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);

    for (int k = 0; k < K; k++) {
        if (k != i && k != j) {
            Eigen::Matrix3d R_qk_h = R_h * (*R_ks)[k].transpose();
            Eigen::Vector3d T_qk_h = -R_qk_h * ((*R_ks)[k] * c_h + (*T_ks)[k]);
            double T_angular_diff = functions::getAngleBetween(T_qk_h, (*T_qks)[k]);
            double R_angular_diff = functions::rotationDifference(R_qk_h, (*R_qks)[k]);
            if (T_angular_diff <= threshold && R_angular_diff <= threshold) {
                indices.push_back(k);
                score += R_angular_diff + T_angular_diff;
            }
        }
    }

    auto t = make_tuple(i, j, score, indices);

    mtx.lock();
    results->push_back(t);
//    cout << i << ", " << j << " finished." << endl;
    mtx.unlock();
}

void findRelativePoses(const string * anchor,
                       const vector<cv::Point2d> * pts_q,
                       const vector<cv::Point2d> * pts_i,
                       const vector<double> * K_q,
                       const vector<double> * K_i,
                       map<string, pair<Eigen::Matrix3d, Eigen::Vector3d>> * rel_poses) {
    try {
        cv::Mat K_q_mat = (cv::Mat_<double>(3, 3) << (*K_q)[2], 0., (*K_q)[0], 0., (*K_q)[3], (*K_q)[1], 0., 0., 1.);
        cv::Mat K_i_mat = (cv::Mat_<double>(3, 3) << (*K_i)[2], 0., (*K_i)[0], 0., (*K_i)[3], (*K_i)[1], 0., 0., 1.);

        vector<cv::Point2d> u_pts_q, u_pts_i;
        cv::Mat d;
        cv::undistortPoints(*pts_q, u_pts_q, K_q_mat, d);
        cv::undistortPoints(*pts_i, u_pts_i, K_i_mat, d);

        double thresh = .5 / (((*K_q)[2] + (*K_q)[3] + (*K_i)[2] + (*K_i)[3]) / 4);

        cv::Mat mask;
        cv::Mat E_ = cv::findEssentialMat(u_pts_i, u_pts_q,
                                          1., cv::Point2d(0., 0.),
                                          cv::RANSAC, 0.9999999999999999, thresh, mask);

        cv::Mat E = E_(cv::Range(0, 3), cv::Range(0, 3));

        cv::Mat R_, T_;
        recoverPose(E, u_pts_i, u_pts_q, R_, T_, 1.0, cv::Point2d(0, 0), mask);

        Eigen::Matrix3d R_qk;
        Eigen::Vector3d T_qk;
        cv2eigen(R_, R_qk);
        cv2eigen(T_, T_qk);

        auto pose = make_pair(R_qk, T_qk);
        auto p = make_pair(*anchor, pose);

        mtx.lock();
        rel_poses->insert(p);
        mtx.unlock();
        return;
    } catch (...) {
        return;
    }
}


int main() {

    string scene = "chess/";
    string dataset = "seven_scenes/";
    string point = "SP";
    string fn = "inliers";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/"+dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/";

    ofstream error;
    error.open(home_dir+scene+point+"_error.txt");

    vector<string> queries = functions::getQueries(home_dir+"q.txt", scene);

    int start = 0;
    for (int q = start; q < queries.size(); q++) {

        string query = queries[q];

        string line = to_string(q); line += "/" + to_string(queries.size())query;

        string file = fn; file += "_" + point;
        auto info = functions::parseRelposeFile(home_dir, query, file);
        auto R_q = get<1>(info);
        auto T_q = get<2>(info);
        auto K_q = get<3>(info);
        auto anchors = get<4>(info);
        auto R_is = get<5>(info);
        auto T_is = get<6>(info);
        auto R_qis = get<7>(info);
        auto T_qis = get<8>(info);
        auto K_is = get<9>(info);
        auto inliers_q = get<10>(info);
        auto inliers_i = get<11>(info);

        int K = int(anchors.size());

        Eigen::Vector3d c_q = -R_q.transpose() * T_q;

        for (int i = 0; i < K; i++) {
            Eigen::Matrix3d R_qi_real = R_q * R_is[i].transpose();
            Eigen::Vector3d T_qi_real = T_q - R_qi_real * T_is[i];
            T_qi_real.normalize();
            double R_error = functions::rotationDifference(R_qi_real, R_qis[i]);
            double T_error = functions::getAngleBetween(T_qi_real, T_qis[i]);
            cout << R_error << ",  " << T_error << endl;
            int check = 0;
        }

        // ALL K -------------------------------------------------------------------------------------------------------

        int count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                count++;
            }
        }
        double threshold = 5.;
        std::thread threads_1[count];
        count = 0;
        auto * results = new vector<tuple<int,int,double,vector<int>>> ();
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                threads_1[count] = thread(findInliers, threshold, i, j, &R_is, &T_is, &R_qis, &T_qis, results);
                this_thread::sleep_for(std::chrono::microseconds (1));
                count++;
            }
        }

        // Sort by number of inliers
        for (auto & th : threads_1) th.join();
        sort(results->begin(), results->end(), [](const auto & a, const auto & b) {
            return get<3>(a).size() > get<3>(b).size();
        });

        // Sort by lowest score
        auto best_set = results->at(0);
        int size = int(get<3>(best_set).size());
        double best_score = get<2>(best_set);
        int idx = 0;
        while (true) {
            try {
                auto set = results->at(idx);
                if (get<3>(set).size() != size) break;
                if (get<2>(set) < best_score) {
                    best_score = get<2>(set);
                    best_set = set;
                }
            } catch (...) {
                break;
            }
            idx++;
        }
        delete results;

        vector<string> best_anchors;
        vector<Eigen::Matrix3d> best_R_is, best_R_qis;
        vector<Eigen::Vector3d> best_T_is, best_T_qis;
        vector<vector<double>> best_K_is;
        vector<vector<cv::Point2d>> best_inliers_q, best_inliers_i;
        for (const auto & i : get<3>(best_set)) {
            best_anchors.push_back(anchors[i]);
            best_R_is.push_back(R_is[i]);
            best_R_qis.push_back(R_qis[i]);
            best_T_is.push_back(T_is[i]);
            best_T_qis.push_back(T_qis[i]);
            best_inliers_q.push_back(inliers_q[i]);
            best_inliers_i.push_back(inliers_i[i]);
            best_K_is.push_back(K_is[i]);
        }

        vector<Eigen::Matrix3d> rotations(best_R_is.size());
        for (int i = 0; i < best_R_is.size(); i++) {
            rotations[i] = best_R_qis[i] * best_R_is[i];
        }

        Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
        Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);

        double c_error_estimation_all = functions::getDistBetween(c_q, c_estimation);
        double R_error_estimation_all = functions::rotationDifference(R_q, R_estimation);

        Eigen::Matrix3d R_adjustment = R_estimation;
        Eigen::Vector3d T_adjustment = - R_estimation * c_estimation;

        pose::adjustHypothesis (best_R_is, best_T_is, best_K_is, K_q, best_inliers_q, best_inliers_i, 5., R_adjustment, T_adjustment);
        Eigen::Vector3d c_adjustment = -R_adjustment.transpose() * T_adjustment;

        double c_error_adjustment_all = functions::getDistBetween(c_q, c_adjustment);
        double R_error_adjustment_all = functions::rotationDifference(R_q, R_adjustment);

        double d = R_adjustment.determinant();

        int stop = 0;

        line += " All_Pre_Adj " + to_string(R_error_estimation_all)
             + " " + to_string(c_error_estimation_all)
             + " All_Post_Adj " + to_string(R_error_adjustment_all)
             + " " + to_string(c_error_adjustment_all);
        //--------------------------------------------------------------------------------------------------------------




        cout << "Finding Zhou solution..." << endl;
        // Zhou Spacing ------------------------------------------------------------------------------------------------
        K = int(anchors.size());
        vector<Eigen::Vector3d> centers (K);
        for(int i = 0; i < K; i++) {
            centers[i] = - R_is[i].transpose() * T_is[i];
        }
        vector<int> spaced_indices = functions::optimizeSpacingZhou(centers, 0.05, 10., 20);
        vector<string> anchors_zhou;
        vector<Eigen::Matrix3d> R_is_zhou, R_qis_zhou;
        vector<Eigen::Vector3d> T_is_zhou, T_qis_zhou;
        vector<vector<double>> K_is_zhou;
        vector<vector<cv::Point2d>> inliers_q_zhou, inliers_i_zhou;
        for (const auto & i : spaced_indices) {
            R_qis_zhou.push_back(R_qis[i]);
            T_qis_zhou.push_back(T_qis[i]);
            anchors_zhou.push_back(anchors[i]);
            R_is_zhou.push_back(R_is[i]);
            T_is_zhou.push_back(T_is[i]);
            K_is_zhou.push_back(K_is[i]);
            inliers_q_zhou.push_back(inliers_q[i]);
            inliers_i_zhou.push_back(inliers_i[i]);
        }

        K = int(anchors_zhou.size());
        count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                count++;
            }
        }
        threshold = 5.;
        std::thread threads_2[count];
        count = 0;
        results = new vector<tuple<int,int,double,vector<int>>> ();
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                threads_2[count] = thread(findInliers, threshold, i, j, &R_is_zhou, &T_is_zhou, &R_qis_zhou, &T_qis_zhou, results);
                this_thread::sleep_for(std::chrono::microseconds (1));
                count++;
            }
        }
        // Sort by number of inliers
        for (auto & th : threads_2) th.join();
        sort(results->begin(), results->end(), [](const auto & a, const auto & b) {
            return get<3>(a).size() > get<3>(b).size();
        });
        // Sort by lowest score
        best_set = results->at(0);
        size = int(get<3>(best_set).size());
        best_score = get<2>(best_set);
        idx = 0;
        while (true) {
            try {
                auto set = results->at(idx);
                if (get<3>(set).size() != size) break;
                if (get<2>(set) < best_score) {
                    best_score = get<2>(set);
                    best_set = set;
                }
            } catch (...) {
                break;
            }
            idx++;
        }
        delete results;

        best_anchors.clear();
        best_R_is.clear();
        best_R_qis.clear();
        best_T_is.clear();
        best_T_qis.clear();
        best_K_is.clear();
        best_inliers_q.clear();
        best_inliers_i.clear();
        for (const auto & i : get<3>(best_set)) {
            best_anchors.push_back(anchors_zhou[i]);
            best_R_is.push_back(R_is_zhou[i]);
            best_R_qis.push_back(R_qis_zhou[i]);
            best_T_is.push_back(T_is_zhou[i]);
            best_T_qis.push_back(T_qis_zhou[i]);
            best_inliers_q.push_back(inliers_q_zhou[i]);
            best_inliers_i.push_back(inliers_i_zhou[i]);
            best_K_is.push_back(K_is_zhou[i]);
        }

        rotations.clear();
        rotations.reserve(best_anchors.size());
        for (int i = 0; i < best_R_is.size(); i++) {
            rotations.emplace_back(best_R_qis[i] * best_R_is[i]);
        }

        c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
        R_estimation = pose::R_q_average(rotations);

        double c_error_estimation_zhou = functions::getDistBetween(c_q, c_estimation);
        double R_error_estimation_zhou = functions::rotationDifference(R_q, R_estimation);

        R_adjustment = R_estimation;
        T_adjustment = - R_estimation * c_estimation;

        pose::adjustHypothesis (best_R_is, best_T_is, best_K_is, K_q, best_inliers_q, best_inliers_i, 5., R_adjustment, T_adjustment);
        c_adjustment = -R_adjustment.transpose() * T_adjustment;

        double c_error_adjustment_zhou = functions::getDistBetween(c_q, c_adjustment);
        double R_error_adjustment_zhou = functions::rotationDifference(R_q, R_adjustment);

        stop = 0;

        line += " Zhou_Pre_Adj " + to_string(R_error_estimation_zhou)
                + " " + to_string(c_error_estimation_zhou)
                + " Zhou_Post_Adj " + to_string(R_error_adjustment_zhou)
                + " " + to_string(c_error_adjustment_zhou);
        //--------------------------------------------------------------------------------------------------------------




        cout << "Finding Our Solution..." << endl;
        // Our Spacing -------------------------------------------------------------------------------------------------
        vector<int> our_indices = functions::optimizeSpacing(c_q, centers, 20, false);
        vector<string> anchors_ours;
        vector<Eigen::Matrix3d> R_is_ours, R_qis_ours;
        vector<Eigen::Vector3d> T_is_ours, T_qis_ours;
        vector<vector<double>> K_is_ours;
        vector<vector<cv::Point2d>> inliers_q_ours, inliers_i_ours;
        for (const auto & i : our_indices) {
            R_qis_ours.push_back(R_qis[i]);
            T_qis_ours.push_back(T_qis[i]);
            anchors_ours.push_back(anchors[i]);
            R_is_ours.push_back(R_is[i]);
            T_is_ours.push_back(T_is[i]);
            K_is_ours.push_back(K_is[i]);
            inliers_q_ours.push_back(inliers_q[i]);
            inliers_i_ours.push_back(inliers_i[i]);
        }

        K = int(anchors_ours.size());
        count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                count++;
            }
        }
        threshold = 5.;
        std::thread threads_3[count];
        count = 0;
        results = new vector<tuple<int,int,double,vector<int>>> ();
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                threads_3[count] = thread(findInliers, threshold, i, j, &R_is_ours, &T_is_ours, &R_qis_ours, &T_qis_ours, results);
                this_thread::sleep_for(std::chrono::microseconds (1));
                count++;
            }
        }
        // Sort by number of inliers
        for (auto & th : threads_3) th.join();
        sort(results->begin(), results->end(), [](const auto & a, const auto & b) {
            return get<3>(a).size() > get<3>(b).size();
        });
        // Sort by lowest score
        best_set = results->at(0);
        size = int(get<3>(best_set).size());
        best_score = get<2>(best_set);
        idx = 0;
        while (true) {
            try {
                auto set = results->at(idx);
                if (get<3>(set).size() != size) break;
                if (get<2>(set) < best_score) {
                    best_score = get<2>(set);
                    best_set = set;
                }
            } catch (...) {
                break;
            }
            idx++;
        }
        delete results;

        best_anchors.clear();
        best_R_is.clear();
        best_R_qis.clear();
        best_T_is.clear();
        best_T_qis.clear();
        best_K_is.clear();
        best_inliers_q.clear();
        best_inliers_i.clear();
        for (const auto & i : get<3>(best_set)) {
            best_anchors.push_back(anchors_ours[i]);
            best_R_is.push_back(R_is_ours[i]);
            best_R_qis.push_back(R_qis_ours[i]);
            best_T_is.push_back(T_is_ours[i]);
            best_T_qis.push_back(T_qis_ours[i]);
            best_inliers_q.push_back(inliers_q_ours[i]);
            best_inliers_i.push_back(inliers_i_ours[i]);
            best_K_is.push_back(K_is_ours[i]);
        }

        rotations.clear();
        rotations.reserve(best_anchors.size());
        for (int i = 0; i < best_R_is.size(); i++) {
            rotations.emplace_back(best_R_qis[i] * best_R_is[i]);
        }

        c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
        R_estimation = pose::R_q_average(rotations);

        double c_error_estimation_ours = functions::getDistBetween(c_q, c_estimation);
        double R_error_estimation_ours = functions::rotationDifference(R_q, R_estimation);

        R_adjustment = R_estimation;
        T_adjustment = - R_estimation * c_estimation;

        pose::adjustHypothesis (best_R_is, best_T_is, best_K_is, K_q, inliers_q, inliers_i, 5., R_adjustment, T_adjustment);
        c_adjustment = -R_adjustment.transpose() * T_adjustment;

        double c_error_adjustment_ours = functions::getDistBetween(c_q, c_adjustment);
        double R_error_adjustment_ours = functions::rotationDifference(R_q, R_adjustment);

        stop = 0;

        line += " Ours_Pre_Adj: " + to_string(R_error_estimation_ours)
                + " " + to_string(c_error_estimation_ours)
                + " Ours_Post_Adj: " + to_string(R_error_adjustment_ours)
                + " " + to_string(c_error_adjustment_ours);

        line += "\n";
        cout << line;
        error << line;
    }
    error.close();
    return 0;
}