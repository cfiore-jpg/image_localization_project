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
                  vector<tuple<
                          int, int, double,
                          vector<int>,
                          vector<Eigen::Matrix3d>,
                          vector<Eigen::Vector3d>,
                          vector<Eigen::Matrix3d>,
                          vector<Eigen::Vector3d>>> * results) {

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
                set_R_ks.push_back((*R_ks)[k]);
                set_T_ks.push_back((*T_ks)[k]);
                set_R_qks.push_back((*R_qks)[k]);
                set_T_qks.push_back((*T_qks)[k]);
                indices.push_back(k);
                score += R_angular_diff + T_angular_diff;
            }
        }
    }

    auto t = make_tuple(i, j, score, indices, set_R_ks, set_T_ks, set_R_qks, set_T_qks);

    mtx.lock();
    results->push_back(t);
//    cout << i << ", " << j << " finished." << endl;
    mtx.unlock();
}


int main() {

    string scene = "chess";
    string data_dir = "/Users/cameronfiore/C++/image_localization_project/data/";
    string scene_dir = data_dir + scene + "/";

    auto rel_poses = functions::getRelativePoses(scene_dir, scene_dir);

    ofstream all_khat_error;
    all_khat_error.open(scene_dir + "all_khat_error.txt");

    for (const auto & query : rel_poses) {

        string query_name = query.first;

        cout << query_name << endl;

        Eigen::Matrix3d R_q; Eigen::Vector3d T_q;
        sevenScenes::getAbsolutePose(query_name, R_q, T_q);
        Eigen::Vector3d c_q = -R_q.transpose() * T_q;

        vector<string> anchor_names;
        vector<Eigen::Matrix3d> R_ks, R_qks;
        vector<Eigen::Vector3d> T_ks, T_qks;

        for (const auto & anchor : query.second) {

            string anchor_name = anchor.first;

            Eigen::Matrix3d R_k, R_qk; Eigen::Vector3d T_k, T_qk;
            sevenScenes::getAbsolutePose(anchor_name, R_k, T_k);
            R_qk = anchor.second.first;
            T_qk = anchor.second.second;

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
        }

        int K = int(R_ks.size());

        auto * results = new vector<tuple<
                int, int, double,
                vector<int>,
                vector<Eigen::Matrix3d>,
                vector<Eigen::Vector3d>,
                vector<Eigen::Matrix3d>,
                vector<Eigen::Vector3d>>> ();

        int count = 0;
        for (int i = 0; i < K - 1; i++) {
            for (int j = i + 1; j < K; j++) {
                count++;
            }
        }

        double threshold = 5.;

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
        tuple<int, int, double,vector<int>,
                vector<Eigen::Matrix3d>,
                vector<Eigen::Vector3d>,
                vector<Eigen::Matrix3d>,
                vector<Eigen::Vector3d>> best_set = results->at(0);
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

        auto best_R_ks = get<4>(best_set);
        auto best_T_ks = get<5>(best_set);
        auto best_R_qks = get<6>(best_set);
        auto best_T_qks = get<7>(best_set);

        vector<Eigen::Matrix3d> rotations(best_R_ks.size());
        for (int i = 0; i < best_R_ks.size(); i++) {
            rotations[i] = best_R_qks[i] * best_R_ks[i];
        }

        Eigen::Vector3d c_est = pose::c_q_closed_form(best_R_ks, best_T_ks, best_R_qks, best_T_qks);
        Eigen::Matrix3d R_est = pose::R_q_average(rotations);

        double c_ = functions::getDistBetween(c_q, c_est);
        double r_ = functions::rotationDifference(R_q, R_est);

        string p = query_name;
        p += "  " + to_string(r_) + "  " + to_string(c_) + "\n";
        cout << p;
        all_khat_error << p;
    }
    all_khat_error.close();

    return 0;
}