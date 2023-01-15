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

//    vector<string> scenes = {"chess/", "fire/", "heads/", "office/", "pumpkin/", "redkitchen/", "stairs/"};
//   vector<string> scenes = {"stairs/"};
//   string dataset = "seven_scenes/";

//    vector<string> scenes = {"GreatCourt/", "KingsCollege/", "OldHospital/", "ShopFacade/", "StMarysChurch/"};
//     // vector<string> scenes = {"OldHospital/"};
//     string dataset = "cambridge/";

    vector<string> scenes = {"query/"};
    string dataset = "aachen/";

    string relpose_file = "relpose_SP";

    // string error_file = "error_SP_justransac";
//    string error_file = "Aachen_eval_MultiLoc";

    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/" + dataset;
    string dir = home_dir;

    double angle_thresh = 5;
    double covis = 2;
    double pixel_thresh = 2.5;
    double cauchy = 1;
    while (angle_thresh <= 20) {
        while (covis <= 10) {
            while (pixel_thresh <= 20) {
                while (cauchy <= 20) {
                    for (const auto & scene: scenes) {
                        ostringstream ef;
                        ef.precision(3);
                        ef << angle_thresh << "_" << covis << "_" << pixel_thresh << "_" << cauchy;
                        string error_file = "Aachen_eval_";
                        error_file += ef.str();
                        ofstream error;
                        error.open(dir + scene + "hp/" + error_file + ".txt");

                        int start = 0;
                        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
//                        vector<string> queries_subset = {queries[11], queries[40], queries[46], queries[57], queries[102]};
                        vector<string> queries_subset = {queries[0]};
                        for (int q = start; q < queries_subset.size(); q++) {
                            cout << "ANGLE: " << angle_thresh << "COVIS: " << covis << "PIXEL: " << pixel_thresh << "CAUCHY: " << cauchy;

                            string query = queries_subset[q];

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

                            vector<thread> threads;
                            vector<tuple<int, int, double, vector<int>>> results;
                            for (int i = 0; i < K - 1; i++) {
                                for (int j = i + 1; j < K; j++) {
                                    threads.emplace_back(
                                            thread(findInliers, angle_thresh, i, j, &R_is, &T_is, &R_qis, &T_qis,
                                                   &inliers_q, &results));
                                    this_thread::sleep_for(std::chrono::microseconds(1));
                                }
                            }
                            for (auto &th: threads) th.join();
                            sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
                                return get<3>(a).size() > get<3>(b).size();
                            });
                            vector<tuple<int, int, double, vector<int>>> results_trimmed;
                            for (int i = 0; i < results.size(); i++) {
                                if (get<3>(results[i]).size() == get<3>(results[0]).size()) {
                                    results_trimmed.push_back(results[i]);
                                }
                            }
                            sort(results_trimmed.begin(), results_trimmed.end(), [](const auto &a, const auto &b) {
                                return get<2>(a) > get<2>(b);
                            });
                            tuple<int, int, double, vector<int>> best_set = results_trimmed[0];


//            cv::Mat q_im = cv::imread(dir+query);
//            for (const auto & i : get<3>(best_set)) {
////            for (int i = 0; i < K; i++) {
//                string anchor = dir+anchors[i].substr(anchors[i].find("db/"));
//                cv::Mat db_im = cv::imread(anchor);
//                int rows = max(q_im.rows, db_im.rows);
//                int cols = q_im.cols + db_im.cols;
//                cv::Mat3b im (rows, cols, cv::Vec3b(0, 0, 0));
//                q_im.copyTo(im(cv::Rect(0, 0, q_im.cols, q_im.rows)));
//                db_im.copyTo(im(cv::Rect(q_im.cols, 0, db_im.cols, db_im.rows)));
//                vector<cv::Point2d> points_q = inliers_q[i];
//                vector<cv::Point2d> points_i = inliers_i[i];
//                for (int j = 0; j < points_q.size(); j++) {
//                    cv::Point2d pt1 = points_q[j];
//                    cv::Point2d pt2 (points_i[j].x + q_im.cols, points_i[j].y);
//                    cv::line(im, pt1, pt2, cv::Scalar(0, 255, 0), 1);
//                }
//                cv::imshow(anchor+"   Index: "+to_string(i)+"   Num Points: "+ to_string(points_q.size()), im);
//                cv::waitKey(0);
//            }


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
//            Eigen::Quaterniond q_adj = Eigen::Quaterniond(R_estimation);
//
//
//
//            string title = "INCLUDED";
//            cv::Mat im = cv::imread(dir + query);
//            vector<double> v_GT, v_EST, v_ADJ;
//            for (int i = 0; i < adj_points.first.size(); i++) {
//
//                cv::Point2d pt = adj_points.first[i];
//                cv::Point2d pt_ud = pose::undistort_point(pt, (K_q[2] + K_q[3]) / 2, K_q[0], K_q[1], K_q[4]);
//                Eigen::Vector3d point3d = adj_points.second[i];
//
////                cv::Point2d reprojGT = pose::reproject3Dto2D(point3d, R_q, T_q, K_q);
//                cv::Point2d reprojEST = pose::reproject3Dto2D(point3d, R_estimation, T_estimation, K_q);
//                cv::Point2d reprojADJ = pose::reproject3Dto2D(point3d, R_adjustment, T_adjustment, K_q);
//
//                cv::circle(im, pt, 6, cv::Scalar(0, 255, 0));
//                cv::circle(im, pt_ud, 6, cv::Scalar(255, 0, 0), -1);
//
////                v_GT.push_back(sqrt(pow(pt.x - reprojGT.x, 2.) + pow(pt.y - reprojGT.y, 2.)));
////                cv::line(im, pt, reprojGT, cv::Scalar(255, 0, 0), 1);
//
//                v_EST.push_back(sqrt(pow(pt_ud.x - reprojEST.x, 2.) + pow(pt_ud.y - reprojEST.y, 2.)));
//                cv::circle(im, reprojEST, 6, cv::Scalar(0, 0, 255), -1);
//                cv::line(im, pt_ud, reprojEST, cv::Scalar(0, 0, 255), 1);
//
//                v_ADJ.push_back(sqrt(pow(pt_ud.x - reprojADJ.x, 2.) + pow(pt_ud.y - reprojADJ.y, 2.)));
//                cv::circle(im, reprojADJ, 6, cv::Scalar(255, 0, 255), -1);
//                cv::line(im, pt_ud, reprojADJ, cv::Scalar(255, 0, 255), 1);
//            }
//            cv::imshow(title, im);
//            cv::waitKey(0);
//            auto msEST = functions::mean_and_stdv(v_EST);
//            auto msADJ = functions::mean_and_stdv(v_ADJ);

                            auto pos = query.find('/');
                            string name = query;
                            while (pos != string::npos) {
                                name = name.substr(pos + 1);
                                pos = name.find('/');
                            }
                            error << name << setprecision(17) << " " << q_adj.w() << " " << q_adj.x() << " "
                                  << q_adj.y() << " " <<
                                  q_adj.z() << " " << T_adjustment[0] << " " << T_adjustment[1] << " "
                                  << T_adjustment[2] << endl;
//            error << name << setprecision(17) << " " << q_adj.w() << " " << q_adj.x() << " " << q_adj.y() << " " <<
//                  q_adj.z() << " " << T_estimation[0] << " " << T_estimation[1] << " " << T_estimation[2] << endl;

                            //  string line;
                            //  line += query + " All_Pre_Adj " + to_string(R_error_estimation_all)
                            //          + " " + to_string(c_error_estimation_all)
                            //          + " All_Post_Adj " + to_string(R_error_adjustment_all)
                            //          + " " + to_string(c_error_adjustment_all);
                            //   error << line << endl;
                            //   cout << line << endl;
                        }
                        error.close();
                    }
                    cauchy += 2.5;
                }
                pixel_thresh += 2.5;
            }
            covis += 1;
        }
        angle_thresh += 2.5;
    }
    return 0;
}