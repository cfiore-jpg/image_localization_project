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
                  vector<tuple<int, int, double, vector<int>>> * results) {

    int K = int((*R_ks).size());
    vector<Eigen::Matrix3d> set_R_ks{(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks{(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks{(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks{(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};

    Eigen::Matrix3d R_q = pose::R_q_average(vector<Eigen::Matrix3d>{(*R_qks)[i] * (*R_ks)[i], (*R_qks)[j] * (*R_ks)[j]});
    Eigen::Vector3d c_q = pose::c_q_closed_form(set_R_ks, set_T_ks, set_R_qks, set_T_qks);

    double r_score = functions::rotationDifference(R_q*(*R_ks)[i].transpose(), (*R_qks)[i]) +
                     functions::rotationDifference(R_q*(*R_ks)[j].transpose(), (*R_qks)[j]);

    double c_score = functions::getAngleBetween(-(*R_qks)[i]*((*R_ks)[i]*c_q+(*T_ks)[i]), (*T_qks)[i]) +
                     functions::getAngleBetween(-(*R_qks)[j]*((*R_ks)[j]*c_q+(*T_ks)[j]), (*T_qks)[j]);

    for (int k = 0; k < K; k++) {
        if (k != i && k != j) {
            double rot_diff = functions::rotationDifference(R_q*(*R_ks)[k].transpose(), (*R_qks)[k]);
            double center_diff = functions::getAngleBetween(-(*R_qks)[k]*((*R_ks)[k]*c_q+(*T_ks)[k]), (*T_qks)[k]);
            if (rot_diff <= thresh && center_diff <= thresh) {
                indices.push_back(k);
                r_score += rot_diff;
                c_score += center_diff;
            }
        }
    }

    double score = r_score * c_score / double(indices.size() * indices.size());

    auto t = make_tuple(i, j, score, indices);

    mtx.lock();
    results->push_back(t);
    mtx.unlock();
}



int main() {

//    pose::solution_tester();
//
//
//
//    exit(0);
    double thresh = 5;

//    vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
//    string dataset = "seven_scenes/";
//    string error_file = "error_SP_SFM";
//    int cutoff = -1;

//     vector<string> scenes = {"GreatCourt/", "KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
     vector<string> scenes = {"KingsCollege/"};
     string dataset = "cambridge/";
     string error_file = "error_fast";
     int cutoff = -1;

    // vector<string> scenes = {"query/"};
    // string dataset = "aachen/";
    // string error_file = "Aachen_eval_MultiLoc";
    // int cutoff = 3;

//    vector<string> scenes = {"query/"};
//    string dataset = "robotcar/";
//    string error_file = "Robotcar_eval_MultiLoc";
//    int cutoff = 2;

    string relpose_file = "relpose_SP_all";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = home_dir;

    for (const auto &scene: scenes) {
        ofstream error;
        error.open(dir + scene + error_file + ".txt");

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {

            cout << q + 1 << "/" << queries.size();
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
            auto pts_q = get<10>(info);
            auto pts_i = get<11>(info);
            int K = int(anchors.size());

            Eigen::Matrix3d R_adjustment;
            Eigen::Vector3d T_adjustment;
            double c_error_est;
            double r_error_est;
            double c_error_adj;
            double r_error_adj;
            double c_error_num;
            double r_error_num;

            if (K == 0) {
                R_adjustment << 1., 0., 0., 0., 1., 0., 0., 0., 1.;
                T_adjustment << 0., 0., 0.;
                cout << endl;
            } else if (K == 1) {
                R_adjustment = R_is[0];
                T_adjustment = T_is[0];
                cout << endl;
            } else {

                //// For speed testing lower K to top 10 ///
                K = 10;
                vector<Eigen::Matrix3d> R_is_top10(10);
                vector<Eigen::Vector3d> T_is_top10(10);
                vector<Eigen::Matrix3d> R_qis_top10(10);
                vector<Eigen::Vector3d> T_qis_top10(10);
                for (int i = 0; i < 10; i++) {
                    R_is_top10[i] = R_is[i];
                    T_is_top10[i] = T_is[i];
                    R_qis_top10[i] = R_qis[i];
                    T_qis_top10[i] = T_qis[i];
                }
                ///////////////////////////////////////////


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

                        //// UNDO THIS CHANGE ////
                        threads[idx] = thread(findInliers, thresh, i, j,
                                              &R_is_top10, &T_is_top10, &R_qis_top10, &T_qis_top10,
                                              &results);
                        idx++;
                    }
                }
                for (auto &th: threads) {
                    th.join();
                }
                sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
                    if (get<3>(a).size() == get<3>(b).size()) {
                        return get<2>(a) < get<2>(b);
                    } else {
                        return get<3>(a).size() > get<3>(b).size();
                    }
                });

                tuple<int, int, double, vector<int>> best_set = results[0];

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
                    best_inliers_q.push_back(pts_q[i]);
                    best_inliers_i.push_back(pts_i[i]);
                    best_K_is.push_back(K_is[i]);
                }
                cout << " " << best_R_is.size() << "/" << R_is.size() << endl;


                vector<Eigen::Matrix3d> rotations(best_R_is.size());
                for (int r = 0; r < best_R_is.size(); r++) {
                    rotations[r] = best_R_qis[r] * best_R_is[r];
                }
                Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
                Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);
                Eigen::Vector3d T_estimation = -R_estimation * c_estimation;
                c_error_est = functions::getDistBetween(c_q, c_estimation);
                r_error_est = functions::rotationDifference(R_q, R_estimation);


                functions::filter_points(15., K_q, R_estimation, T_estimation,
                                         K_is, R_is, T_is,
                                         pts_q, pts_i);


//                vector<pair<pair<double,double>,vector<pair<int,int>>>> all_matches = functions::findSharedMatches(2, best_R_is, best_T_is, best_K_is, best_inliers_q, best_inliers_i);
//                cv::Mat qIm = cv::imread(dir+query);
//                for (int i = 0; i < all_matches.size(); i++) {
//                    auto p = all_matches[i];
//                    cv::Point2d pt2D(p.first.first, p.first.second);
//                    Eigen::Vector3d pt3D = pose::nview(p.second, best_R_is, best_T_is, best_K_is, best_inliers_i);
//                    cv::Point2d reprojected2Dpt = pose::reproject3Dto2D(pt3D, R_estimation, T_estimation, K_q);
//
//
//                    cv::circle(qIm, pt2D, 5., {0,0,255}, -1);
//                    cv::circle(qIm, reprojected2Dpt, 5., {0,255,0});
//
//
//                    int pause = 0;
//
//                }
//                cv::imshow("final-pose-level filtering", qIm);
//                cv::waitKey();


                Eigen::Matrix3d R_adjusted = R_estimation;
                Eigen::Vector3d T_adjusted = -R_estimation * c_estimation;
                auto adj_points = pose::adjustHypothesis(R_is, T_is, K_is,
                                                         K_q, pts_q, pts_i,
                                                         R_adjusted, T_adjusted);
                Eigen::Vector3d c_adjusted = -R_adjusted.transpose() * T_adjusted;
                c_error_adj = functions::getDistBetween(c_q, c_adjusted);
                r_error_adj = functions::rotationDifference(R_q, R_adjusted);

                cout << c_error_adj << endl;
                cout << r_error_adj << endl;

                exit(0);

                int stop = 0;

//                Eigen::Matrix3d R_numerical = R_q;
//                Eigen::Vector3d T_numerical = T_q;
//                auto numerical_points = pose::num_sys_solution(best_R_is, best_T_is, best_K_is, K_q, best_inliers_q,
//                                                               best_inliers_i,
//                                                               R_numerical, T_numerical);
//                Eigen::Vector3d c_numerical = -R_numerical.transpose() * T_numerical;
//                c_error_num = functions::getDistBetween(c_q, c_numerical);
//                r_error_num = functions::rotationDifference(R_q, R_numerical);

//                cout << " " << adj_points.first.size() << endl;
            }

            if (dataset == "aachen/" || dataset == "robotcar/") {
                Eigen::Quaterniond q_adj = Eigen::Quaterniond(R_adjustment);
                auto pos = query.find('/');
                string name = query;
                for (int c = 0; c < cutoff; c++) {
                    name = name.substr(pos + 1);
                    pos = name.find('/');
                }
                error << name << setprecision(17) << " " << q_adj.w() << " " << q_adj.x() << " "
                      << q_adj.y() << " " <<
                      q_adj.z() << " " << T_adjustment[0] << " " << T_adjustment[1] << " "
                      << T_adjustment[2] << endl;
            } else {
                string line;
                line += query + " Pre_Adj " + to_string(r_error_est)
                        + " " + to_string(c_error_est)
                        + " Post_Adj " + to_string(r_error_adj)
                        + " " + to_string(c_error_adj);
                error << line << endl;
                cout << line << endl;
            }
        }
        error.close();
    }
    return 0;
}