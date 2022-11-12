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
                  vector<tuple<int, int, double, vector<pair<int, double>>>> * results) {

    int K = int((*R_ks).size());

    vector<Eigen::Matrix3d> set_R_ks {(*R_ks)[i], (*R_ks)[j]};
    vector<Eigen::Vector3d> set_T_ks {(*T_ks)[i], (*T_ks)[j]};
    vector<Eigen::Matrix3d> set_R_qks {(*R_qks)[i], (*R_qks)[j]};
    vector<Eigen::Vector3d> set_T_qks {(*T_qks)[i], (*T_qks)[j]};
    vector<pair<int, double>> indices {make_pair(i, 0.), make_pair(j, 0.)};
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
                indices.emplace_back(k, R_angular_diff + T_angular_diff);
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

    vector<string> scenes = {"chess/"};

    string dataset = "seven_scenes/";
    string relpose_file = "relpose_SIFT";
    string error_file1 = "error_K_sweep_by_support";
    string error_file2 = "error_K_sweep_by_spacing";


    string ccv_dir = "/users/cfiore/data/cfiore/image_localization_project/data/" + dataset;
    string home_dir = "/Users/cameronfiore/C++/image_localization_project/data/";
    string dir = home_dir;

    for (const auto &scene: scenes) {

        ofstream error1;
        error1.open(dir + scene + error_file1 + ".txt");

        ofstream error2;
        error2.open(dir + scene + error_file2 + ".txt");

        double threshold = 10.;
        double adj_threshold = 3.;

        int start = 0;
        vector<string> queries = functions::getQueries(dir + "q.txt", scene);
        vector<Eigen::Matrix3d> R_is_subset, R_qis_subset, rotations_subset;
        vector<Eigen::Vector3d> T_is_subset, T_qis_subset;
        for (int q = start; q < queries.size(); q++) {

            cout << q + 1 << "/" << queries.size() << "..." << endl;
            string query = queries[q];
            string line1 = query;
            string line2 = query;


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
            auto *results = new vector<tuple<int, int, double, vector<pair<int, double>>>>();
            for (int i = 0; i < K - 1; i++) {
                for (int j = i + 1; j < K; j++) {
                    threads_1[count] = thread(findInliers, threshold, i, j, &R_is, &T_is, &R_qis, &T_qis, results);
                    this_thread::sleep_for(std::chrono::microseconds(1));
                    count++;
                }
            }
            for (auto &th: threads_1) th.join();

            // Sort by number of inliers
            sort(results->begin(), results->end(), [](const auto &a, const auto &b) {
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



            int I_i = get<0>(best_set);
            int I_j = get<1>(best_set);
            vector<Eigen::Matrix3d> base_R_is {R_is[I_i], R_is[I_j]};
            vector<Eigen::Matrix3d> base_R_qis {R_qis[I_i], R_qis[I_j]};
            vector<Eigen::Matrix3d> base_rotations {R_qis[I_i] * R_is[I_i], R_qis[I_j] * R_is[I_j]};
            vector<Eigen::Vector3d> base_T_is {T_is[I_i], T_is[I_j]};
            vector<Eigen::Vector3d> base_T_qis {T_qis[I_i], T_qis[I_j]};


            vector<pair<int, double>> best_indices = get<3>(best_set);
            sort(best_indices.begin(), best_indices.end(), [](const auto &a, const auto &b) {
                return a.second < b.second;
            });

            vector<Eigen::Matrix3d> best_R_is, best_R_qis;
            vector<Eigen::Vector3d> best_T_is, best_T_qis;
            vector<vector<cv::Point2d>> best_inliers_q, best_inliers_i;
            for (auto & best_index : best_indices) {
                int id = best_index.first;
                if (id != I_i && id != I_j) {
                    best_R_is.push_back(R_is[id]);
                    best_R_qis.push_back(R_qis[id]);
                    best_T_is.push_back(T_is[id]);
                    best_T_qis.push_back(T_qis[id]);
                    best_inliers_q.push_back(inliers_q[id]);
                    best_inliers_i.push_back(inliers_i[id]);
                }
            }

            vector<Eigen::Matrix3d> rotations(best_R_is.size());
            for (int i = 0; i < best_R_is.size(); i++) {
                rotations[i] = best_R_qis[i] * best_R_is[i];
            }

            K = int(best_R_is.size());
            vector<Eigen::Vector3d> centers(K);
            for (int i = 0; i < K; i++) {
                centers[i] = -best_R_is[i].transpose() * best_T_is[i];
            }

            for(int k = 0; k <= K; k++) {

                R_is_subset = base_R_is;
                R_qis_subset = base_R_qis;
                rotations_subset = base_rotations;
                T_is_subset = base_T_is;
                T_qis_subset = base_T_qis;
                for (int i = 0; i < k; i++) {
                    R_is_subset.push_back(best_R_is[i]);
                    T_is_subset.push_back(best_T_is[i]);
                    R_qis_subset.push_back(best_R_qis[i]);
                    T_qis_subset.push_back(best_T_qis[i]);
                    rotations_subset.push_back(rotations[i]);
                }
                Eigen::Vector3d c_estimation = pose::c_q_closed_form(R_is_subset, T_is_subset, R_qis_subset, T_qis_subset);
                Eigen::Matrix3d R_estimation = pose::R_q_average(rotations_subset);
                Eigen::Vector3d T_estimation = -R_estimation * c_estimation;
                double c_error = functions::getDistBetween(c_q, c_estimation);
                double R_error = functions::rotationDifference(R_q, R_estimation);
                line1 += " " + to_string(k) + " " + to_string(c_error) + " " + to_string(R_error);
                R_is_subset.clear();
                T_is_subset.clear();
                R_qis_subset.clear();
                T_qis_subset.clear();
                rotations_subset.clear();

                R_is_subset = base_R_is;
                R_qis_subset = base_R_qis;
                rotations_subset = base_rotations;
                T_is_subset = base_T_is;
                T_qis_subset = base_T_qis;
                vector<int> subset_indices = functions::optimizeSpacing(c_q, centers, k, false);
                for (const auto & i: subset_indices) {
                    R_is_subset.push_back(best_R_is[i]);
                    T_is_subset.push_back(best_T_is[i]);
                    R_qis_subset.push_back(best_R_qis[i]);
                    T_qis_subset.push_back(best_T_qis[i]);
                    rotations_subset.push_back(rotations[i]);
                }
                c_estimation = pose::c_q_closed_form(R_is_subset, T_is_subset, R_qis_subset, T_qis_subset);
                R_estimation = pose::R_q_average(rotations_subset);
                T_estimation = -R_estimation * c_estimation;
                c_error = functions::getDistBetween(c_q, c_estimation);
                R_error = functions::rotationDifference(R_q, R_estimation);
                line2 += " " + to_string(k) + " " + to_string(c_error) + " " + to_string(R_error);
                R_is_subset.clear();
                T_is_subset.clear();
                R_qis_subset.clear();
                T_qis_subset.clear();
                rotations_subset.clear();
            }
            error1 << line1 << endl;
            error2 << line2 << endl;

//            Eigen::Vector3d c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
//            Eigen::Matrix3d R_estimation = pose::R_q_average(rotations);
//            Eigen::Vector3d T_estimation = -R_estimation * c_estimation;
//
//            double c_error_estimation_all = functions::getDistBetween(c_q, c_estimation);
//            double R_error_estimation_all = functions::rotationDifference(R_q, R_estimation);


//            line += " All_Pre_Adj " + to_string(R_error_estimation_all) + " " + to_string(c_error_estimation_all);

//            Eigen::Matrix3d R_adjustment = R_estimation;
//            Eigen::Vector3d T_adjustment = -R_estimation * c_estimation;
//
//            double avg_rep = pose::adjustHypothesis(best_R_is, best_T_is, best_K_is, K_q, best_inliers_q,
//                                                    best_inliers_i, adj_threshold, R_adjustment, T_adjustment);
//            Eigen::Vector3d c_adjustment = -R_adjustment.transpose() * T_adjustment;
//
//            double c_error_adjustment_all = functions::getDistBetween(c_q, c_adjustment);
//            double R_error_adjustment_all = functions::rotationDifference(R_q, R_adjustment);
//
//            line += " All_Avg_Rep " + to_string(avg_rep)
//                    + " All_Post_Adj " + to_string(R_error_adjustment_all)
//                    + " " + to_string(c_error_adjustment_all);

            //--------------------------------------------------------------------------------------------------------------














            // Zhou Spacing ------------------------------------------------------------------------------------------------
//            K = int(anchors.size());
//            vector<Eigen::Vector3d> centers(K);
//            for (int i = 0; i < K; i++) {
//                centers[i] = -R_is[i].transpose() * T_is[i];
//            }
//            vector<int> spaced_indices = functions::optimizeSpacingZhou(centers, 0.05, 10., 20);
//            vector<string> anchors_zhou;
//            vector<Eigen::Matrix3d> R_is_zhou, R_qis_zhou;
//            vector<Eigen::Vector3d> T_is_zhou, T_qis_zhou;
//            vector<vector<double>> K_is_zhou;
//            vector<vector<cv::Point2d>> inliers_q_zhou, inliers_i_zhou;
//            for (const auto &i: spaced_indices) {
//                R_qis_zhou.push_back(R_qis[i]);
//                T_qis_zhou.push_back(T_qis[i]);
//                anchors_zhou.push_back(anchors[i]);
//                R_is_zhou.push_back(R_is[i]);
//                T_is_zhou.push_back(T_is[i]);
//                K_is_zhou.push_back(K_is[i]);
//                inliers_q_zhou.push_back(inliers_q[i]);
//                inliers_i_zhou.push_back(inliers_i[i]);
//            }
//
//            K = int(anchors_zhou.size());
//            count = 0;
//            for (int i = 0; i < K - 1; i++) {
//                for (int j = i + 1; j < K; j++) {
//                    count++;
//                }
//            }
//            std::thread threads_2[count];
//            count = 0;
//            results = new vector<tuple<int, int, double, vector<int>>>();
//            for (int i = 0; i < K - 1; i++) {
//                for (int j = i + 1; j < K; j++) {
//                    threads_2[count] = thread(findInliers, threshold, i, j, &R_is_zhou, &T_is_zhou, &R_qis_zhou,
//                                              &T_qis_zhou, results);
//                    this_thread::sleep_for(std::chrono::microseconds(1));
//                    count++;
//                }
//            }
//            // Sort by number of inliers
//            for (auto &th: threads_2) th.join();
//            sort(results->begin(), results->end(), [](const auto &a, const auto &b) {
//                return get<3>(a).size() > get<3>(b).size();
//            });
//            // Sort by lowest score
//            best_set = results->at(0);
//            size = int(get<3>(best_set).size());
//            best_score = get<2>(best_set);
//            idx = 0;
//            while (true) {
//                try {
//                    auto set = results->at(idx);
//                    if (get<3>(set).size() != size) break;
//                    if (get<2>(set) < best_score) {
//                        best_score = get<2>(set);
//                        best_set = set;
//                    }
//                } catch (...) {
//                    break;
//                }
//                idx++;
//            }
//            delete results;
//
//            best_anchors.clear();
//            best_R_is.clear();
//            best_R_qis.clear();
//            best_T_is.clear();
//            best_T_qis.clear();
//            best_K_is.clear();
//            best_inliers_q.clear();
//            best_inliers_i.clear();
//            for (const auto &i: get<3>(best_set)) {
//                best_anchors.push_back(anchors_zhou[i]);
//                best_R_is.push_back(R_is_zhou[i]);
//                best_R_qis.push_back(R_qis_zhou[i]);
//                best_T_is.push_back(T_is_zhou[i]);
//                best_T_qis.push_back(T_qis_zhou[i]);
//                best_inliers_q.push_back(inliers_q_zhou[i]);
//                best_inliers_i.push_back(inliers_i_zhou[i]);
//                best_K_is.push_back(K_is_zhou[i]);
//            }
//
//            rotations.clear();
//            rotations.reserve(best_anchors.size());
//            for (int i = 0; i < best_R_is.size(); i++) {
//                rotations.emplace_back(best_R_qis[i] * best_R_is[i]);
//            }
//
//            c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
//            R_estimation = pose::R_q_average(rotations);
//
//            double c_error_estimation_zhou = functions::getDistBetween(c_q, c_estimation);
//            double R_error_estimation_zhou = functions::rotationDifference(R_q, R_estimation);
//
//            line += " Zhou_Pre_Adj " + to_string(R_error_estimation_zhou) + " " + to_string(c_error_estimation_zhou);
//
//            // R_adjustment = R_estimation;
//            // T_adjustment = - R_estimation * c_estimation;
//
//            // avg_rep = pose::adjustHypothesis(best_R_is, best_T_is, best_K_is, K_q, best_inliers_q, best_inliers_i, adj_threshold, R_adjustment, T_adjustment);
//            // c_adjustment = -R_adjustment.transpose() * T_adjustment;
//
//            // double c_error_adjustment_zhou = functions::getDistBetween(c_q, c_adjustment);
//            // double R_error_adjustment_zhou = functions::rotationDifference(R_q, R_adjustment);
//
//
//            // line += " All_Avg_Rep " + to_string(avg_rep)
//            //         + " Zhou_Pre_Adj " + to_string(R_error_estimation_zhou)
//            //         + " " + to_string(c_error_estimation_zhou)
//            //         + " Zhou_Post_Adj " + to_string(R_error_adjustment_zhou)
//            //         + " " + to_string(c_error_adjustment_zhou);
//            //--------------------------------------------------------------------------------------------------------------
//
//
//
//
//
//
//
//
//
//
//            // Our Spacing -------------------------------------------------------------------------------------------------
//            vector<int> our_indices = functions::optimizeSpacing(c_q, centers, 20, false);
//            vector<string> anchors_ours;
//            vector<Eigen::Matrix3d> R_is_ours, R_qis_ours;
//            vector<Eigen::Vector3d> T_is_ours, T_qis_ours;
//            vector<vector<double>> K_is_ours;
//            vector<vector<cv::Point2d>> inliers_q_ours, inliers_i_ours;
//            for (const auto &i: our_indices) {
//                R_qis_ours.push_back(R_qis[i]);
//                T_qis_ours.push_back(T_qis[i]);
//                anchors_ours.push_back(anchors[i]);
//                R_is_ours.push_back(R_is[i]);
//                T_is_ours.push_back(T_is[i]);
//                K_is_ours.push_back(K_is[i]);
//                inliers_q_ours.push_back(inliers_q[i]);
//                inliers_i_ours.push_back(inliers_i[i]);
//            }
//
//            K = int(anchors_ours.size());
//            count = 0;
//            for (int i = 0; i < K - 1; i++) {
//                for (int j = i + 1; j < K; j++) {
//                    count++;
//                }
//            }
//            std::thread threads_3[count];
//            count = 0;
//            results = new vector<tuple<int, int, double, vector<int>>>();
//            for (int i = 0; i < K - 1; i++) {
//                for (int j = i + 1; j < K; j++) {
//                    threads_3[count] = thread(findInliers, threshold, i, j, &R_is_ours, &T_is_ours, &R_qis_ours,
//                                              &T_qis_ours, results);
//                    this_thread::sleep_for(std::chrono::microseconds(1));
//                    count++;
//                }
//            }
//            // Sort by number of inliers
//            for (auto &th: threads_3) th.join();
//            sort(results->begin(), results->end(), [](const auto &a, const auto &b) {
//                return get<3>(a).size() > get<3>(b).size();
//            });
//            // Sort by lowest score
//            best_set = results->at(0);
//            size = int(get<3>(best_set).size());
//            best_score = get<2>(best_set);
//            idx = 0;
//            while (true) {
//                try {
//                    auto set = results->at(idx);
//                    if (get<3>(set).size() != size) break;
//                    if (get<2>(set) < best_score) {
//                        best_score = get<2>(set);
//                        best_set = set;
//                    }
//                } catch (...) {
//                    break;
//                }
//                idx++;
//            }
//            delete results;
//
//            best_anchors.clear();
//            best_R_is.clear();
//            best_R_qis.clear();
//            best_T_is.clear();
//            best_T_qis.clear();
//            best_K_is.clear();
//            best_inliers_q.clear();
//            best_inliers_i.clear();
//            for (const auto &i: get<3>(best_set)) {
//                best_anchors.push_back(anchors_ours[i]);
//                best_R_is.push_back(R_is_ours[i]);
//                best_R_qis.push_back(R_qis_ours[i]);
//                best_T_is.push_back(T_is_ours[i]);
//                best_T_qis.push_back(T_qis_ours[i]);
//                best_inliers_q.push_back(inliers_q_ours[i]);
//                best_inliers_i.push_back(inliers_i_ours[i]);
//                best_K_is.push_back(K_is_ours[i]);
//            }
//
//            rotations.clear();
//            for (int i = 0; i < best_R_is.size(); i++) {
//                rotations.emplace_back(best_R_qis[i] * best_R_is[i]);
//            }
//
//            c_estimation = pose::c_q_closed_form(best_R_is, best_T_is, best_R_qis, best_T_qis);
//            R_estimation = pose::R_q_average(rotations);
//
//            double c_error_estimation_ours = functions::getDistBetween(c_q, c_estimation);
//            double R_error_estimation_ours = functions::rotationDifference(R_q, R_estimation);
//
//            line += " Ours_Pre_Adj " + to_string(R_error_estimation_ours) + " " + to_string(c_error_estimation_ours);
//
//            // R_adjustment = R_estimation;
//            // T_adjustment = - R_estimation * c_estimation;
//
//            // avg_rep = pose::adjustHypothesis(best_R_is, best_T_is, best_K_is, K_q, best_inliers_q, best_inliers_i, adj_threshold, R_adjustment, T_adjustment);
//            // c_adjustment = -R_adjustment.transpose() * T_adjustment;
//
//            // double c_error_adjustment_ours = functions::getDistBetween(c_q, c_adjustment);
//            // double R_error_adjustment_ours = functions::rotationDifference(R_q, R_adjustment);
//
//            // line += " All_Avg_Rep " + to_string(avg_rep)
//            //         + " Ours_Pre_Adj " + to_string(R_error_estimation_ours)
//            //         + " " + to_string(c_error_estimation_ours)
//            //         + " Ours_Post_Adj " + to_string(R_error_adjustment_ours)
//            //         + " " + to_string(c_error_adjustment_ours);
//
//            line += "\n";
//            cout << line << endl;
//            error << line;
        }
        error1.close();
        error2.close();
    }
    return 0;
}