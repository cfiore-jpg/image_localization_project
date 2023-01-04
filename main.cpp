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

//    vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
//     vector<string> scenes = {"stairs/"};
//    string dataset = "seven_scenes/";

//   vector<string> scenes = {"KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
    vector<string> scenes = {"KingsCollege/"};
    string dataset = "cambridge/";


    string relpose_file = "relpose_SP";
    string error_file = "error_SP_justransac";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = home_dir;

    for (const auto &scene: scenes) {
        ofstream error;
        error.open(dir + scene + error_file + ".txt");

        double threshold = 3.;
        double adj_threshold = 50.;

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        for (int q = start; q < queries.size(); q++) {

            cout << q + 1 << "/" << queries.size() << "..." << endl;

            string query = queries[q];
            string line = query;

            auto info = functions::parseRelposeFile(dir, query, relpose_file);
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

            //        for (int i = 0; i < K; i++) {
            //            Eigen::Matrix3d R_qi_real = R_q * R_is[i].transpose();
            //            Eigen::Vector3d T_qi_real = T_q - R_qi_real * T_is[i];
            //            T_qi_real.normalize();
            //            double R_error = functions::rotationDifference(R_qi_real, R_qis[i]);
            //            double T_error = functions::getAngleBetween(T_qi_real, T_qis[i]);
            //            cout << R_error << ",  " << T_error << endl;
            //            int check = 0;
            //        }

            // ALL K -------------------------------------------------------------------------------------------------------
            int count = 0;
            for (int i = 0; i < K - 1; i++) {
                for (int j = i + 1; j < K; j++) {
                    count++;
                }
            }

            std::thread threads_1[count];
            count = 0;
            vector<tuple<int, int, double, vector<int>>> results;
            for (int i = 0; i < K - 1; i++) {
                for (int j = i + 1; j < K; j++) {
                    threads_1[count] = thread(findInliers, threshold, i, j, &R_is, &T_is, &R_qis, &T_qis, &results);
                    this_thread::sleep_for(std::chrono::microseconds(1));
                    count++;
                }
            }

            // Sort by number of inliers
            for (auto &th: threads_1) th.join();
            sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
                return get<3>(a).size() > get<3>(b).size();
            });

            // Sort by lowest score
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

            Eigen::Matrix3d R_adjustment = R_estimation;
            Eigen::Vector3d T_adjustment = -R_estimation * c_estimation;
            pose::adjustHypothesis(best_R_is, best_T_is, best_K_is, K_q, best_inliers_q, best_inliers_i, adj_threshold, R_adjustment, T_adjustment);
        //    pose::adjustHypothesis(R_is, T_is, K_is, K_q, inliers_q, inliers_i, adj_threshold, R_adjustment, T_adjustment);
            Eigen::Vector3d c_adjustment = -R_adjustment.transpose() * T_adjustment;
            double c_error_adjustment_all = functions::getDistBetween(c_q, c_adjustment);
            double R_error_adjustment_all = functions::rotationDifference(R_q, R_adjustment);

              auto r = functions::findSharedMatches(best_R_is, best_T_is, best_K_is, best_inliers_q, best_inliers_i);
//            auto r = functions::findSharedMatches(R_is, T_is, K_is, inliers_q, inliers_i);

            // string title;
            // cv::Mat im;
            // cv::Mat inc = cv::imread(dir + query);
            // cv::Mat ninc = cv::imread(dir + query);
            // for (const auto & p: r) {

            //     if (p.second.size() < 2) break;
            //     cv::Point2d pt(p.first.first, p.first.second);
            //     auto point3d = pose::get3DPoint(p.second);

            //     cv::Point2d reproj2DGT = pose::reproject3Dto2D(point3d, R_q, T_q, K_q);
            //     cv::Point2d reproj2DEST = pose::reproject3Dto2D(point3d, R_estimation, T_estimation, K_q);
            //     cv::Point2d reproj2DAdj = pose::reproject3Dto2D(point3d, R_adjustment, T_adjustment, K_q);

            //     if (sqrt(pow(pt.x - reproj2DEST.x, 2.) + pow(pt.y - reproj2DEST.y, 2.)) > adj_threshold) {
            //         title = "NOT INCLUDED";
            //         auto color = cv::Scalar(255, 0, 0);
            //         cv::circle(ninc, pt, 5, color, -1);
            //         cv::imshow(title, ninc);
            //         cv::waitKey(0);
            //         color = cv::Scalar(0, 255, 0);
            //         cv::circle(ninc, reproj2DGT, 5, color, -1);
            //         cv::imshow(title, ninc);
            //         cv::waitKey(0);
            //         color = cv::Scalar(0, 0, 255);
            //         cv::circle(ninc, reproj2DEST, 5, color, -1);
            //         cv::imshow(title, ninc);
            //         cv::waitKey(0);
            //         color = cv::Scalar(255, 0, 255);
            //         cv::circle(ninc, reproj2DAdj, 5, color, -1);
            //         cv::imshow(title, ninc);
            //         cv::waitKey(0);

            //     } else {
            //         title = "INCLUDED";
            //         auto color = cv::Scalar(255, 0, 0);
            //         cv::circle(inc, pt, 5, color, -1);
            //         cv::imshow(title, inc);
            //         cv::waitKey(0);
            //         color = cv::Scalar(0, 255, 0);
            //         cv::circle(inc, reproj2DGT, 5, color, -1);
            //         cv::imshow(title, inc);
            //         cv::waitKey(0);
            //         color = cv::Scalar(0, 0, 255);
            //         cv::circle(inc, reproj2DEST, 5, color, -1);
            //         cv::imshow(title, inc);
            //         cv::waitKey(0);
            //         color = cv::Scalar(255, 0, 255);
            //         cv::circle(inc, reproj2DAdj, 5, color, -1);
            //         cv::imshow(title, inc);
            //         cv::waitKey(0);
            //     }
            // }

            line += " All_Pre_Adj " + to_string(R_error_estimation_all)
                    + " " + to_string(c_error_estimation_all)
                    + " All_Post_Adj " + to_string(R_error_adjustment_all)
                    + " " + to_string(c_error_adjustment_all);

            error << line << endl;
            cout << line << endl;
            //--------------------------------------------------------------------------------------------------------------
        }
        error.close();
    }
    return 0;
}