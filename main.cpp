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

    vector<Eigen::Matrix3d> set_R_ks{(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks{(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks{(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks{(*T_qks)[i], (*T_qks)[j]};
    vector<int> indices{i, j};
    double score = 0;

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

//    vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
//   vector<string> scenes = {"stairs/"};
//   string dataset = "seven_scenes/";

//    vector<string> scenes = {"KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
    vector<string> scenes = {"KingsCollege/"};
    string dataset = "cambridge/";

//    vector<string> scenes = {"query/"};
//    string dataset = "aachen/";

    string relpose_file = "relpose_SP_r";
    string error_file = "error_SP_justransac";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = home_dir;

    double angle_thresh = 5;
    double covis = 3;
    double pixel_thresh = 3;
    double post_ransac = -1;
    double reproj_tolerance = 100000;

    for (const auto &scene: scenes) {
        ofstream error;
        error.open(dir + scene + error_file + ".txt");

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {
            cout << q + 1 << "/" << queries.size() << " ";

            string query = queries[q];

            auto info = functions::parseRelposeFile(dir, query, relpose_file, 1);
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
            Eigen::Vector3d c_q = -R_q.transpose() * T_q;
            int K = int(anchors.size());

            vector<thread> threads;
            vector<tuple<int, int, double, vector<int>>> results;
            for (int i = 0; i < K - 1; i++) {
                for (int j = i + 1; j < K; j++) {
                    threads.emplace_back(
                            thread(findInliers, angle_thresh, i, j, &R_is, &T_is, &R_qis, &T_qis, &results));
                    this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
            for (auto &th: threads) th.join();
            sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
                return get<3>(a).size() > get<3>(b).size();
            });
            auto best_set = results.at(0);
            int size = int(get<3>(best_set).size());
            double best_score = get<2>(best_set);
            int idx = 0;
            while (true) {
                try {
                    auto set = results.at(idx);
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

            Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
            Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);
            Eigen::Vector3d T_estimation = -R_estimation * c_estimation;
            double c_error_estimation_all = functions::getDistBetween(c_q, c_estimation);
            double R_error_estimation_all = functions::rotationDifference(R_q, R_estimation);

            cout << best_R_is.size() << "/" << R_is.size() << " ";

            Eigen::Matrix3d R_adjustment = R_estimation;
            Eigen::Vector3d T_adjustment = -R_estimation * c_estimation;
            auto adj_points = pose::adjustHypothesis(best_R_is,
                                                     best_T_is,
                                                     best_K_is,
                                                     K_q,
                                                     best_inliers_q,
                                                     best_inliers_i,
                                                     covis,
                                                     pixel_thresh,
                                                     post_ransac,
                                                     reproj_tolerance,
                                                     R_adjustment,
                                                     T_adjustment);
            Eigen::Vector3d c_adjustment = -R_adjustment.transpose() * T_adjustment;
            double c_error_adjustment_all = functions::getDistBetween(c_q, c_adjustment);
            double R_error_adjustment_all = functions::rotationDifference(R_q, R_adjustment);

            cout << adj_points.first.size() << endl;

//            Eigen::Quaterniond q_adj = Eigen::Quaterniond(R_adjustment);
//
//
//            string title = "INCLUDED";
//            cv::Mat im = cv::imread(dir + query);
//            vector<double> v_GT, v_EST, v_ADJ;
//            for (int i = 0; i < adj_points.first.size(); i++) {
//
//                cv::Point2d pt = adj_points.first[i];
//                Eigen::Vector3d point3d = adj_points.second[i];
////
////                cv::Point2d reprojGT = pose::reproject3Dto2D(point3d, R_q, T_q, K_q);
//                cv::Point2d reprojEST = pose::reproject3Dto2D(point3d, R_estimation, T_estimation, K_q);
//                cv::Point2d reprojADJ = pose::reproject3Dto2D(point3d, R_adjustment, T_adjustment, K_q);
//
//                cv::circle(im, pt, 3, cv::Scalar(0, 0, 0));
//
////                v_GT.push_back(sqrt(pow(pt.x - reprojGT.x, 2.) + pow(pt.y - reprojGT.y, 2.)));
//                auto color = cv::Scalar(255, 0, 0);
////                cv::circle(im, reprojGT, 3, color, -1);
////                cv::line(im, pt, reprojGT, color, 1);
//
//                v_EST.push_back(sqrt(pow(pt.x - reprojEST.x, 2.) + pow(pt.y - reprojEST.y, 2.)));
//                color = cv::Scalar(0, 0, 255);
//                cv::circle(im, reprojEST, 3, color, -1);
//                cv::line(im, pt, reprojEST, color, 1);
//
//                v_ADJ.push_back(sqrt(pow(pt.x - reprojADJ.x, 2.) + pow(pt.y - reprojADJ.y, 2.)));
//                color = cv::Scalar(255, 0, 255);
//                cv::circle(im, reprojADJ, 3, color, -1);
//                cv::line(im, pt, reprojADJ, color, 1);
//            }
//            auto msEST = functions::mean_and_stdv(v_EST);
//            auto msADJ = functions::mean_and_stdv(v_EST);
//
//
//            cv::imshow(title, im);
//            cv::waitKey(0);
//
//            auto pos = query.find('/');
//            string name = query;
//            while (pos != string::npos) {
//                name = name.substr(pos + 1);
//                pos = name.find('/');
//            }
//            error << name << setprecision(17) << " " << q_adj.w() << " " << q_adj.x() << " " << q_adj.y() << " " <<
//                  q_adj.z() << " " << T_adjustment[0] << " " << T_adjustment[1] << " " << T_adjustment[2] << endl;

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