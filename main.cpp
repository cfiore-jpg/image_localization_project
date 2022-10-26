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

    string scene = "chess";
    string dataset = "seven_scenes/";
    string relpose_fn = "relpose_SIFT";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/"+dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/";

    vector<string> queries = functions::getQueries(home_dir+"q.txt", scene);

    int start = 0;

    for (int q = start; q < queries.size(); q++) {

        string query = queries[q];

        cout << query;

        auto info = functions::parseRelposeFile(home_dir, query, relpose_fn);
        auto R_q = get<1>(info);
        auto T_q = get<2>(info);
        auto K_q = get<3>(info);
        auto anchors = get<4>(info);
        auto R_is = get<5>(info);
        auto T_is = get<6>(info);
        auto K_is = get<7>(info);
        auto all_pts_q = get<8>(info);
        auto all_pts_i = get<9>(info);

        int K = int(anchors.size());

        Eigen::Vector3d c_q = -R_q.transpose() * T_q;

        auto m = new map<string, pair<Eigen::Matrix3d, Eigen::Vector3d>> ();
        std::thread threads_0[K];
        int a = 0;
        for (int i = 0; i < K; i++) {
            threads_0[i] = thread(findRelativePoses, &anchors[i], &all_pts_q[i], &all_pts_i[i], &K_q, &K_is[i], m);
            this_thread::sleep_for(std::chrono::microseconds (1));
        }
        for (auto & th : threads_0) th.join();

        vector<string> anchors_calc;
        vector<Eigen::Matrix3d> R_is_calc, R_qis_calc;
        vector<Eigen::Vector3d> T_is_calc, T_qis_calc;
        vector<vector<double>> K_is_calc;
        vector<vector<cv::Point2d>> all_pts_q_calc, all_pts_i_calc;
        for (int i = 0; i < K; i++) {
            try {
                auto pose = m->at(anchors[i]);
                R_qis_calc.push_back(pose.first);
                T_qis_calc.push_back(pose.second);
                anchors_calc.push_back(anchors[i]);
                R_is_calc.push_back(R_is[i]);
                T_is_calc.push_back(T_is[i]);
                K_is_calc.push_back(K_is[i]);
                all_pts_q_calc.push_back(all_pts_q[i]);
                all_pts_i_calc.push_back(all_pts_i[i]);
            } catch (...) {
                continue;
            }
            Eigen::Matrix3d R_qi_real = R_q * R_is_calc[i].transpose();
            Eigen::Vector3d T_qi_real = T_q - R_qi_real * T_is_calc[i];
            T_qi_real.normalize();
            double R_error = functions::rotationDifference(R_qi_real, R_qis_calc[i]);
            double T_error = functions::getAngleBetween(T_qi_real, T_qis_calc[i]);
            int check = 0;
        }
        delete m;


        // RANSAC ------------------------------------------------------------------------------------------------------
        K = int(R_is_calc.size());

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
                threads_1[count] = thread(findInliers, threshold, i, j, &R_is_calc, &T_is_calc, &R_qis_calc, &T_qis_calc, results);
                this_thread::sleep_for(std::chrono::microseconds (1));
                count++;
            }
        }
        for (auto & th : threads_1) th.join();

        sort(results->begin(), results->end(), [](const auto & a, const auto & b) {
            return get<3>(a).size() > get<3>(b).size();
        });

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
        vector<vector<cv::Point2d>> best_all_pts_q, best_all_pts_i;
        for (const auto & i : get<3>(best_set)) {
            best_anchors.push_back(anchors_calc[i]);
            best_R_is.push_back(R_is_calc[i]);
            best_R_qis.push_back(R_qis_calc[i]);
            best_T_is.push_back(T_is_calc[i]);
            best_T_qis.push_back(T_qis_calc[i]);
            best_all_pts_q.push_back(all_pts_q_calc[i]);
            best_all_pts_i.push_back(all_pts_i_calc[i]);
            best_K_is.push_back(K_is_calc[i]);
        }

        vector<Eigen::Matrix3d> rotations(best_R_is.size());
        for (int i = 0; i < best_R_is.size(); i++) {
            rotations[i] = best_R_qis[i] * best_R_is[i];
        }

        Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
        Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);

        double c_error_estimation = functions::getDistBetween(c_q, c_estimation);
        double R_error_estimation = functions::rotationDifference(R_q, R_estimation);

        Eigen::Matrix3d R_adjustment = R_estimation;
        Eigen::Vector3d T_adjustment = - R_estimation * c_estimation;

        pose::adjustHypothesis (best_R_is, best_T_is, best_K_is, K_q, best_all_pts_q, best_all_pts_i, 5., R_adjustment, T_adjustment);
        Eigen::Vector3d c_adjustment = -R_adjustment.transpose() * T_adjustment;

        double c_error_adjustment = functions::getDistBetween(c_q, c_adjustment);
        double R_error_adjustment = functions::rotationDifference(R_q, R_adjustment);

        int stop = 0;

        cout << " (" << R_error_estimation
             << ", " << c_error_estimation
             << ")   (" << R_error_adjustment
             << ", " << c_error_adjustment
             << ")" << endl;




//        auto spaced = functions::optimizeSpacingZhou(anchor_names, 0.05, 10., 20, "7-Scenes");
//        R_ks.clear(); R_qks.clear(); T_ks.clear(); T_qks.clear(); all_points.clear(); K1s.clear(); K2s.clear();
//        for (const auto & sp : spaced) {
//
//            auto anchor = anchors.at(sp);
//
//            string anchor_name = sp;
//
//            Eigen::Matrix3d R_k, R_qk;
//            Eigen::Vector3d T_k, T_qk;
//            sevenScenes::getAbsolutePose(anchor_name, R_k, T_k);
//            R_qk = get<0>(anchor);
//            T_qk = get<1>(anchor);
//
//            vector<double> K1 = get<2>(anchor);
//            vector<double> K2 = get<3>(anchor);
//
//            auto points = get<4>(anchor);
//            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
//            Eigen::Vector3d T_qk_real = T_q - R_qk_real * T_k;
//            T_qk.normalize();
//
//            double r_error = functions::rotationDifference(R_qk, R_qk_real);
//            double t_error = functions::getAngleBetween(T_qk, T_qk_real);
//
////            cout << "      " << anchor_name << "  R Error: " << r_error << "  T Error: " << t_error << endl;
//
//            R_ks.push_back(R_k);
//            T_ks.push_back(T_k);
//            R_qks.push_back(R_qk);
//            T_qks.push_back(T_qk);
//            all_points.push_back(points);
//            K1s.push_back(K1);
//            K2s.push_back(K2);
//        }
//
//        K = int(R_ks.size());
//
//        auto results2 = new vector<tuple<int, int, double, vector<int>>> ();
//
//        count = 0;
//        for (int i = 0; i < K - 1; i++) {
//            for (int j = i + 1; j < K; j++) {
//                count++;
//            }
//        }
//
//        threshold = 10.;
//
//        std::thread threads2[count];
//        count = 0;
//        for (int i = 0; i < K - 1; i++) {
//            for (int j = i + 1; j < K; j++) {
//                threads2[count] = thread(findInliers, threshold, i, j, &R_ks, &T_ks, &R_qks, &T_qks, results2);
//                this_thread::sleep_for(std::chrono::microseconds (1));
//                count++;
//            }
//        }
//        for (auto & th : threads2) th.join();
//
//        sort(results2->begin(), results2->end(), [](const auto & a, const auto & b) {
//            return get<3>(a).size() > get<3>(b).size();
//        });
//
//        s = int(get<3>(results2->at(0)).size());
//        best_score = get<2>(results2->at(0));
//        best_set = results2->at(0);
//        idx = 0;
//        while (true) {
//            try {
//                auto set = results2->at(idx);
//                if (get<3>(set).size() != s) break;
//                if (get<2>(set) < best_score) {
//                    best_score = get<2>(set);
//                    best_set = set;
//                }
//            } catch (...) {
//                break;
//            }
//            idx++;
//        }
//
//        best_anchor_names.clear();
//        best_R_ks.clear(); best_R_qks.clear();
//        best_T_ks.clear(); best_T_qks.clear();
//        best_all_points.clear();
//        best_K1s.clear(); best_K2s.clear();
//        for (const auto & i : get<3>(best_set)) {
//            best_anchor_names.push_back(anchor_names[i]);
//            best_R_ks.push_back(R_ks[i]);
//            best_R_qks.push_back(R_qks[i]);
//            best_T_ks.push_back(T_ks[i]);
//            best_T_qks.push_back(T_qks[i]);
//            best_all_points.push_back(all_points[i]);
//            best_K1s.push_back(K1s[i]);
//            best_K2s.push_back(K2s[i]);
//        }
//
//        vector<Eigen::Matrix3d> rotations_2(best_R_ks.size());
//        for (int i = 0; i < best_R_ks.size(); i++) {
//            rotations_2[i] = best_R_qks[i] * best_R_ks[i];
//        }
//
//        c_est_initial = pose::c_q_closed_form(best_R_ks, best_T_ks, best_R_qks, best_T_qks);
//        R_est_initial = pose::R_q_average(rotations_2);
//
//        double c_spaced = functions::getDistBetween(c_q, c_est_initial);
//        double r_spaced = functions::rotationDifference(R_q, R_est_initial);
//
//        Eigen::Matrix3d r_adj_spaced = R_est_initial;
//        Eigen::Vector3d t_adj_spaced = - R_est_initial * c_est_initial;
//        pose::adjustHypothesis (best_R_ks, best_T_ks, best_all_points, 5., best_K1s, best_K2s, r_adj_spaced, t_adj_spaced);
//        Eigen::Vector3d c_adj_spaced = -r_adj_spaced.transpose() * t_adj_spaced;
//
//        c_spaced = functions::getDistBetween(c_q, c_adj_spaced);
//        r_spaced = functions::rotationDifference(R_q, r_adj_spaced);
//
//        string p = query_name;
//        p += "  " + to_string(r_all) + "  " + to_string(c_all) + "  " + to_string(r_spaced) + "  " + to_string(c_spaced) + "\n";
//        error << p;
//        cout << p;

//        delete results2;
    }

    return 0;
}