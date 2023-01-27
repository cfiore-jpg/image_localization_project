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

void findInliers (double thresh,
                  int i,
                  int j,
                  const vector<Eigen::Matrix3d> * R_ks,
                  const vector<Eigen::Vector3d> * T_ks,
                  const vector<Eigen::Matrix3d> * R_qks,
                  const vector<Eigen::Vector3d> * T_qks,
                  const vector<vector<cv::Point2d>> * match_num,
                  vector<tuple<int, int, double, vector<int>>> * results) {

    int K = int((*R_ks).size());

    vector<Eigen::Matrix3d> set_R_ks{(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks{(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks{(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks{(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};
    int score = int((*match_num)[i].size()) + int((*match_num)[j].size());

    Eigen::Matrix3d R_q = pose::R_q_average(vector<Eigen::Matrix3d>{(*R_qks)[i] * (*R_ks)[i], (*R_qks)[j] * (*R_ks)[j]});
    Eigen::Vector3d c_q = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);

    for (int k = 0; k < K; k++) {
        if (k != i && k != j) {
            double rot_diff = functions::rotationDifference(R_q*(*R_ks)[k].transpose(), (*R_qks)[k]);
            double center_diff = functions::getAngleBetween(-(*R_qks)[k]*((*R_ks)[k]*c_q+(*T_ks)[k]), (*T_qks)[k]);
            if (rot_diff <= thresh && center_diff <= thresh) {
                indices.push_back(k);
                score += int((*match_num)[k].size());
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

//     vector<string> scenes = {"GreatCourt/", "KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
    vector<string> scenes = {"KingsCollege/"};
      string dataset = "cambridge/";
      string error_file = "study_k_sweep";

    string relpose_file = "relpose_SP";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = home_dir;

    double thresh = 5;

    for (const auto &scene: scenes) {
        ofstream error;
        error.open(dir + scene + error_file + ".txt");

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {

            cout << q + 1 << "/" << queries.size() << endl;
            string query = queries[q];

            auto info = functions::parseRelposeFile(dir, query, relpose_file);
            auto R_q = get<1>(info);
            auto T_q = get<2>(info);
            auto K_q = get<3>(info);
            Eigen::Vector3d c_q = -R_q.transpose() * T_q;

            auto anchors = get<4>(info);
            auto R_is = get<5>(info);
            auto T_is = get<6>(info);
            auto R_qis = get<7>(info);
            auto T_qis = get<8>(info);
            auto K_is = get<9>(info);
            auto inliers_q = get<10>(info);
            auto inliers_i = get<11>(info);


            for(int K = 5; K < anchors.size(); K+=5) {
                vector<string> anchors_sub (K);
                vector<Eigen::Matrix3d> R_is_sub (K);
                vector<Eigen::Vector3d> T_is_sub (K);
                vector<Eigen::Matrix3d> R_qis_sub (K);
                vector<Eigen::Vector3d> T_qis_sub (K);
                vector<vector<double>> K_is_sub (K);
                vector<vector<cv::Point2d>> inliers_q_sub (K);
                vector<vector<cv::Point2d>> inliers_i_sub (K);
                for(int i = 0; i < K; i++) {
                    anchors_sub[i] = anchors[i];
                    R_is_sub[i] = R_is[i];
                    T_is_sub[i] = T_is[i];
                    R_qis_sub[i] = R_qis[i];
                    T_qis_sub[i] = T_qis[i];
                    K_is_sub[i] = K_is[i];
                    inliers_q_sub[i] = inliers_q[i];
                    inliers_i_sub[i] = inliers_i[i];
                }

                int s = 0;
                for (int i = 0; i < K - 1; i++) {
                    for (int j = i + 1; j < K; j++) {
                        s++;
                    }
                }
                int idx = 0;
                vector<thread> threads(s);
                vector<tuple<int, int, double, vector<int>>> results;
                for (int i = 0; i < K - 1; i++) {
                    for (int j = i + 1; j < K; j++) {
                        threads[idx] = thread(findInliers, thresh, i, j, &R_is_sub, &T_is_sub, &R_qis_sub, &T_qis_sub, &inliers_q_sub, &results);
                        idx++;
                    }
                }
                for (auto &th: threads) {
                    th.join();
                }
                sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
                    return get<3>(a).size() > get<3>(b).size();
                });
                vector<tuple<int, int, double, vector<int>>> results_trimmed;
                for (int i = 0; i < results.size(); i++) {
                    if (get<3>(results[i]).size() == get<3>(results[0]).size()) {
                        results_trimmed.push_back(results[i]);
                    } else {
                        break;
                    }
                }
                sort(results_trimmed.begin(), results_trimmed.end(), [](const auto &a, const auto &b) {
                    return get<2>(a) > get<2>(b);
                });
                tuple<int, int, double, vector<int>> best_set = results_trimmed[0];

                vector<string> best_anchors;
                vector<Eigen::Matrix3d> best_R_is, best_R_qis;
                vector<Eigen::Vector3d> best_T_is, best_T_qis;
                vector<vector<double>> best_K_is;
                vector<vector<cv::Point2d>> best_inliers_q, best_inliers_i;
                for (const auto &i: get<3>(best_set)) {
                    best_anchors.push_back(anchors_sub[i]);
                    best_R_is.push_back(R_is_sub[i]);
                    best_R_qis.push_back(R_qis_sub[i]);
                    best_T_is.push_back(T_is_sub[i]);
                    best_T_qis.push_back(T_qis_sub[i]);
                    best_inliers_q.push_back(inliers_q_sub[i]);
                    best_inliers_i.push_back(inliers_i_sub[i]);
                    best_K_is.push_back(K_is_sub[i]);
                }
                vector<Eigen::Matrix3d> rotations(best_R_is.size());
                for (int i = 0; i < best_R_is.size(); i++) {
                    rotations[i] = best_R_qis[i] * best_R_is[i];
                }

                Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
                Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);
                Eigen::Vector3d T_estimation = -R_estimation * c_estimation;
                double c_error_est = functions::getDistBetween(c_q, c_estimation);
                double r_error_est = functions::rotationDifference(R_q, R_estimation);


                Eigen::Matrix3d R_adjusted = R_estimation;
                Eigen::Vector3d T_adjusted = -R_estimation * c_estimation;
                auto points = pose::adjustHypothesis(best_R_is, best_T_is, best_K_is, K_q, best_inliers_q, best_inliers_i,
                                                     R_adjusted, T_adjusted);
                Eigen::Vector3d c_adjusted = -R_adjusted.transpose() * T_adjusted;
                double c_error_adj = functions::getDistBetween(c_q, c_adjusted);
                double r_error_adj = functions::rotationDifference(R_q, R_adjusted);

                cout << K << " " << c_error_est << " " << r_error_est << " " << c_error_adj << " " << r_error_adj << " ";
                error << K << " " << c_error_est << " " << r_error_est << " " << c_error_adj << " " << r_error_adj << " ";
            }
            cout << endl;
            error << endl;
        }
        error.close();
    }
    return 0;
}
