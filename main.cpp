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


int main() {

    string scene = "chess";
    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/";
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/";
    string scene_dir = home_dir + scene + "/";

    auto rel_poses = functions::getRelativePoses(scene_dir, scene_dir);

    ofstream error;
    error.open(scene_dir + "SP_all_spaced.txt");

    for (const auto & query : rel_poses) {

        string query_name = query.first;

        Eigen::Matrix3d R_q; Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query_name, R_q, T_q);
        Eigen::Vector3d c_q = -R_q.transpose() * T_q;

        vector<string> anchor_names;
        vector<Eigen::Matrix3d> R_ks, R_qks;
        vector<Eigen::Vector3d> T_ks, T_qks;
        vector<vector<pair<cv::Point2d, cv::Point2d>>> all_points;
        vector<vector<double>> K1s, K2s;

        for (const auto & anchor : query.second) {

            string anchor_name = anchor.first;
            anchor_names.push_back(anchor_name);

            Eigen::Matrix3d R_k, R_qk;
            Eigen::Vector3d T_k, T_qk;
            sevenScenes::getAbsolutePose(anchor_name, R_k, T_k);
            R_qk = get<0>(anchor.second);
            T_qk = get<1>(anchor.second);

            vector<double> K1 = get<2>(anchor.second);
            vector<double> K2 = get<3>(anchor.second);

            auto points = get<4>(anchor.second);
            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
            Eigen::Vector3d T_qk_real = T_q - R_qk_real * T_k;
            T_qk.normalize();

            double r_error = functions::rotationDifference(R_qk, R_qk_real);
            double t_error = functions::getAngleBetween(T_qk, T_qk_real);

//            cout << "      " << anchor_name << "  R Error: " << r_error << "  T Error: " << t_error << endl;

            R_ks.push_back(R_k);
            T_ks.push_back(T_k);
            R_qks.push_back(R_qk);
            T_qks.push_back(T_qk);
            all_points.push_back(points);
            K1s.push_back(K1);
            K2s.push_back(K2);
        }

        int K = int(R_ks.size());

        auto * results = new vector<tuple<int, int, double,vector<int>>> ();

        int count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                count++;
            }
        }

        double threshold = 10.;

        std::thread threads[count];
        count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                threads[count] = thread(findInliers, threshold, i, j, &R_ks, &T_ks, &R_qks, &T_qks, results);
                this_thread::sleep_for(std::chrono::microseconds (1));
                count++;
            }
        }
        for (auto & th : threads) th.join();

        sort(results->begin(), results->end(), [](const auto & a, const auto & b) {
            return get<3>(a).size() > get<3>(b).size();
        });

        int s = int(get<3>(results->at(0)).size());
        double best_score = get<2>(results->at(0));
        tuple<int, int, double,vector<int>> best_set = results->at(0);
        int idx = 0;
        while (true) {
            auto set = results->at(idx);
            idx++;
            if (get<3>(set).size() != s) break;
            if (get<2>(set) < best_score) {
                best_score = get<2>(set);
                best_set = set;
            }
        }

        vector<string> best_anchor_names;
        vector<Eigen::Matrix3d> best_R_ks, best_R_qks;
        vector<Eigen::Vector3d> best_T_ks, best_T_qks;
        vector<vector<pair<cv::Point2d, cv::Point2d>>> best_all_points;
        vector<vector<double>> best_K1s, best_K2s;
        for (const auto & i : get<3>(best_set)) {
            best_anchor_names.push_back(anchor_names[i]);
            best_R_ks.push_back(R_ks[i]);
            best_R_qks.push_back(R_qks[i]);
            best_T_ks.push_back(T_ks[i]);
            best_T_qks.push_back(T_qks[i]);
            best_all_points.push_back(all_points[i]);
            best_K1s.push_back(K1s[i]);
            best_K2s.push_back(K2s[i]);
        }

        vector<Eigen::Matrix3d> rotations(best_R_ks.size());
        for (int i = 0; i < best_R_ks.size(); i++) {
            rotations[i] = best_R_qks[i] * best_R_ks[i];
        }

        Eigen::Vector3d c_est_initial = pose::c_q_closed_form(best_R_ks, best_T_ks, best_R_qks, best_T_qks);
        Eigen::Matrix3d R_est_initial = pose::R_q_average(rotations);

        double c_all = functions::getDistBetween(c_q, c_est_initial);
        double r_all = functions::rotationDifference(R_q, R_est_initial);

        Eigen::Matrix3d r_adj = R_est_initial;
        Eigen::Vector3d t_adj = - R_est_initial * c_est_initial;
        pose::adjustHypothesis (best_R_ks, best_T_ks, best_all_points, 3., best_K1s, best_K2s, r_adj, t_adj);
        Eigen::Vector3d c_adj = -r_adj.transpose() * t_adj;

        c_all = functions::getDistBetween(c_q, c_adj);
        r_all = functions::rotationDifference(R_q, r_adj);




        auto spaced = functions::optimizeSpacingZhou(anchor_names, 0.05, 10., 20, "7-Scenes");
        R_ks.clear(); R_qks.clear(); T_ks.clear(); T_qks.clear(); all_points.clear(); K1s.clear(); K2s.clear();
        for (const auto & sp : spaced) {

            auto anchor = query.second.at(sp);

            string anchor_name = sp;

            Eigen::Matrix3d R_k, R_qk;
            Eigen::Vector3d T_k, T_qk;
            sevenScenes::getAbsolutePose(anchor_name, R_k, T_k);
            R_qk = get<0>(anchor);
            T_qk = get<1>(anchor);

            vector<double> K1 = get<2>(anchor);
            vector<double> K2 = get<3>(anchor);

            auto points = get<4>(anchor);
            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
            Eigen::Vector3d T_qk_real = T_q - R_qk_real * T_k;
            T_qk.normalize();

            double r_error = functions::rotationDifference(R_qk, R_qk_real);
            double t_error = functions::getAngleBetween(T_qk, T_qk_real);

//            cout << "      " << anchor_name << "  R Error: " << r_error << "  T Error: " << t_error << endl;

            R_ks.push_back(R_k);
            T_ks.push_back(T_k);
            R_qks.push_back(R_qk);
            T_qks.push_back(T_qk);
            all_points.push_back(points);
            K1s.push_back(K1);
            K2s.push_back(K2);
        }

        K = int(R_ks.size());

        auto results2 = new vector<tuple<int, int, double, vector<int>>> ();

        count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                count++;
            }
        }

        threshold = 10.;

        count = 0;
        std::thread threads2[count];
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                threads2[count] = thread(findInliers, threshold, i, j, &R_ks, &T_ks, &R_qks, &T_qks, results2);
                this_thread::sleep_for(std::chrono::microseconds (1));
                count++;
            }
        }
        for (auto & th : threads2) th.join();

        sort(results2->begin(), results2->end(), [](const auto & a, const auto & b) {
            return get<3>(a).size() > get<3>(b).size();
        });

        s = int(get<3>(results2->at(0)).size());
        best_score = get<2>(results2->at(0));
        best_set = results2->at(0);
        idx = 0;
        while (true) {
            auto set = results2->at(idx);
            idx++;
            if (get<3>(set).size() != s) break;
            if (get<2>(set) < best_score) {
                best_score = get<2>(set);
                best_set = set;
            }
        }

        best_anchor_names.clear();
        best_R_ks.clear(); best_R_qks.clear();
        best_T_ks.clear(); best_T_qks.clear();
        best_all_points.clear();
        best_K1s.clear(); best_K2s.clear();
        for (const auto & i : get<3>(best_set)) {
            best_anchor_names.push_back(anchor_names[i]);
            best_R_ks.push_back(R_ks[i]);
            best_R_qks.push_back(R_qks[i]);
            best_T_ks.push_back(T_ks[i]);
            best_T_qks.push_back(T_qks[i]);
            best_all_points.push_back(all_points[i]);
            best_K1s.push_back(K1s[i]);
            best_K2s.push_back(K2s[i]);
        }

        vector<Eigen::Matrix3d> rotations_2(best_R_ks.size());
        for (int i = 0; i < best_R_ks.size(); i++) {
            rotations_2[i] = best_R_qks[i] * best_R_ks[i];
        }

        c_est_initial = pose::c_q_closed_form(best_R_ks, best_T_ks, best_R_qks, best_T_qks);
        R_est_initial = pose::R_q_average(rotations_2);

        double c_spaced = functions::getDistBetween(c_q, c_est_initial);
        double r_spaced = functions::rotationDifference(R_q, R_est_initial);

        Eigen::Matrix3d r_adj_spaced = R_est_initial;
        Eigen::Vector3d t_adj_spaced = - R_est_initial * c_est_initial;
        pose::adjustHypothesis (best_R_ks, best_T_ks, best_all_points, 5., best_K1s, best_K2s, r_adj, t_adj);
        Eigen::Vector3d c_adj_spaced = -r_adj.transpose() * t_adj;

        c_spaced = functions::getDistBetween(c_q, c_adj);
        r_spaced = functions::rotationDifference(R_q, r_adj);

        string p = query_name;
        p += "  " + to_string(r_all) + "  " + to_string(c_all) + "  " + to_string(r_spaced) + "  " + to_string(c_spaced) + "\n";
        error << p;
        cout << p;

        delete results;
        delete results2;
    }
    error.close();

    return 0;
}