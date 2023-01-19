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
                  const vector<vector<cv::Point2d>> * match_num,
                  vector<tuple<int, int, double, vector<int>>> * results) {

    int K = int((*R_ks).size());

    vector<Eigen::Matrix3d> set_R_ks{(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks{(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks{(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks{(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};
    int score = int((*match_num)[i].size()) + int((*match_num)[j].size());

    Eigen::Matrix3d R_h = pose::R_q_average(
            vector<Eigen::Matrix3d>{(*R_qks)[i] * (*R_ks)[i], (*R_qks)[j] * (*R_ks)[j]});
    Eigen::Vector3d c_h = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);

    for (int k = 0; k < K; k++) {
        if (k != i && k != j) {
            Eigen::Matrix3d R_qk_h = R_h * (*R_ks)[k].transpose();
            Eigen::Vector3d T_qk_h = -R_qk_h * ((*R_ks)[k] * c_h + (*T_ks)[k]);
            double T_angular_diff = functions::getAngleBetween(T_qk_h, (*T_qks)[k]);
            double R_angular_diff = functions::rotationDifference(R_qk_h, (*R_qks)[k]);
            if (T_angular_diff <= threshold && R_angular_diff <= threshold) {
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

    // vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
//   vector<string> scenes = {"stairs/"};
//    string dataset = "seven_scenes/";

//    vector<string> scenes = {"GreatCourt/", "KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
   vector<string> scenes = {"KingsCollege/"};
   string dataset = "cambridge/";

    // vector<string> scenes = {"query/"};
    // string dataset = "aachen/";

    string relpose_file = "relpose_SP";

    string error_file = "error_SP_justransac";
//    string error_file = "Aachen_eval_MultiLoc";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = ccv_dir;

    double angle_thresh = 5;
    double covis = 2;
    double cauchy = 5;

    for (const auto & scene: scenes) {
        ofstream error;
        error.open(dir + scene + error_file + ".txt");

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {
            cout << q+1 << "/" << queries.size();
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
            int K = int(anchors.size());

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
                    threads[idx] = thread(findInliers, angle_thresh, i, j, &R_is, &T_is, &R_qis, &T_qis, &inliers_q, &results);
                    idx++;
                }
            }
            for (auto & th: threads) {
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

            Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis,
                                                                    best_T_qis);
            Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);
            Eigen::Vector3d T_estimation = -R_estimation * c_estimation;
            double c_error_estimation_all = functions::getDistBetween(c_q, c_estimation);
            double R_error_estimation_all = functions::rotationDifference(R_q, R_estimation);

            cout << " " << best_R_is.size() << "/" << R_is.size() << " ";

            Eigen::Matrix3d R_adjustment = R_estimation;
            Eigen::Vector3d T_adjustment = -R_estimation * c_estimation;
            auto adj_points = pose::adjustHypothesis(best_R_is,
                                                        best_T_is,
                                                        best_K_is,
                                                        K_q,
                                                        best_inliers_q,
                                                        best_inliers_i,
                                                        covis,
                                                        -1,
                                                        -1,
                                                        -1,
                                                        cauchy,
                                                        R_adjustment,
                                                        T_adjustment);
            Eigen::Vector3d c_adjustment = -R_adjustment.transpose() * T_adjustment;
            double c_error_adjustment_all = functions::getDistBetween(c_q, c_adjustment);
            double R_error_adjustment_all = functions::rotationDifference(R_q, R_adjustment);

            cout << adj_points.first.size() << endl;

            Eigen::Quaterniond q_adj = Eigen::Quaterniond(R_adjustment);

            // auto pos = query.find('/');
            // string name = query;
            // while (pos != string::npos) {
            //     name = name.substr(pos + 1);
            //     pos = name.find('/');
            // }
            
            // error << name << setprecision(17) << " " << q_adj.w() << " " << q_adj.x() << " "
            //         << q_adj.y() << " " <<
            //         q_adj.z() << " " << T_adjustment[0] << " " << T_adjustment[1] << " "
            //         << T_adjustment[2] << endl;

             string line;
             line += query + " All_Pre_Adj " + to_string(R_error_estimation_all)
                     + " " + to_string(c_error_estimation_all)
                     + " All_Post_Adj " + to_string(R_error_adjustment_all)
                     + " " + to_string(c_error_adjustment_all);
              error << line << endl;
              cout << line << endl;
        }
        error.close();
    }
    return 0;
}
