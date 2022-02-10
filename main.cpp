
#include "include/sevenScenes.h"
#include "include/functions.h"
#include "include/poseEstimation.h"
#include "include/calibrate.h"
#include "matplotlibcpp.h"
#include <iostream>
#include <Eigen/SVD>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>



#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
#define SCENE 0
#define IMAGE_LIST "/Users/cameronfiore/C++/image_localization_project/data/images_1000.txt"
#define EXT ".color.png"
#define GENERATE_DATABASE false

using namespace std;
using namespace cv;
using namespace chrono;
namespace plt = matplotlibcpp;

int main() {


//      string query = "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-03/frame-000000";

//    string query = "/Users/cameronfiore/C++/image_localization_project/data/chess/seq-03/frame-000400";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/fire/seq-03/frame-000300";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000495";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/office/seq-06/frame-000160";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-07/frame-000490";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-06/frame-000490";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/stairs/seq-01/frame-000260";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/heads/seq-01/frame-000654";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/office/seq-06/frame-000058";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/pumpkin/seq-07/frame-000647";
//    string query = "/Users/cameronfiore/C++/image_localization_project/data/redkitchen/seq-04/frame-000373";


//// Testing Rotation

//    auto retrieved = functions::retrieveSimilar(query, 200, 1.6);
//    auto spaced = functions::optimizeSpacing(retrieved, 10, false);
//
//    Eigen::Vector3d c_q = sevenScenes::getT(query);
//    Eigen::Matrix3d R_q_real;
//    Eigen::Vector3d t_q_real;
//    sevenScenes::getAbsolutePose(query, R_q_real, t_q_real);
//    const double K[4] = {523.538, 529.669, 314.245, 237.595};
//
//    vector<Eigen::Matrix3d> R_k;
//    vector<Eigen::Vector3d> t_k;
//    vector<Eigen::Matrix3d> R_qk;
//    vector<Eigen::Vector3d> t_qk;
//
//    for (const auto & im: spaced) {
//        Eigen::Matrix3d R;
//        Eigen::Vector3d t;
//        Eigen::Matrix3d R_;
//        Eigen::Vector3d t_;
//        sevenScenes::getAbsolutePose(im, R, t);
//        int num = functions::getRelativePose(im, query, K, "SIFT", R_, t_);
////        R_ = R_q_real * R.transpose();
////        t_ = t_q_real - R_ * t;
////        t_.normalize();
//        R_k.push_back(R); t_k.push_back(t); R_qk.push_back(R_); t_qk.push_back(t_);
//    }
//
//    Eigen::Vector3d c_q_guess = pose::hypothesizeQueryCenter(R_k, t_k, R_qk, t_qk);
//    Eigen::Matrix3d R_q_guess = pose::hypothesizeQueryRotation(c_q, R_k, t_k, t_qk);
//
//    double c_error = functions::getDistBetween(c_q, c_q_guess);
//    double r_error = functions::rotationDifference(R_q_real, R_q_guess);
//
//    cout << "done." << endl;





//// Testing Anchor Image Performance
//    vector<string> all;
//    auto info = sevenScenes::createInfoVector();
//    functions::createQueryVector(all, info, 0);
//
//    int max_num = 0;
//    int min_num = 20;
//    int total_recall = 0;
//    int total_images = 0;
//    for (const auto & query : all) {
//
//        string scene = functions::getScene(query);
//
//        auto retrieved = functions::retrieveSimilar(query, 200, 1.6);
//        auto spaced = functions::optimizeSpacing(retrieved, 20, false);
//
//        if (spaced.size() > max_num) max_num = int (spaced.size());
//        if (spaced.size() < min_num) min_num = int (spaced.size());
//
//        total_images += int (spaced.size());
//        for (const auto & s : spaced) {
//            if (functions::getScene(s) == scene) {
//                total_recall++;
//            }
//        }
//    }
//
//    cout << "Max Returned: " << max_num << "/" << "20" << endl;
//    cout << "Min Returned: " << min_num << "/" << "20" << endl;
//    cout << "Recall: " << double (total_recall) / double (total_images) << endl;
//
//
//
//    vector<int> x (20);
//    for (int i = 1; i <= 20; i++) {
//        x[i] = i;
//    }
//    vector<double> max (20);
//    vector<double> min (20);
//    int upper = 0;
//    auto retrieved = functions::retrieveSimilar(query, 200, 1.6);
//    for (int i = 1; i <= 20; i++) {
//        auto spaced = functions::optimizeSpacing(retrieved, i, false);
//        int most = 0;
//        int least = 1000;
//        for (const auto & s : spaced) {
//            vector<cv::Point2d> pts_i, pts_q;
//            functions::findMatches(s, query, "SIFT", 0.8, pts_i, pts_q);
//            if (pts_i.size() > most) most = int (pts_i.size());
//            if (pts_i.size() < least) least = int (pts_i.size());
//        }
//        if (most > upper) upper = most;
//        max[i-1] = most;
//        min[i-1] = least;
//    }
//
//    plt::plot(x, max);
//    plt::plot(x, min);
//    plt::ylim(0, upper + 50);
//    plt::title(query);
//    plt::show();





//// Calibration
//    calibrate::run();







////// Dataset Visualization
//    vector<string> all;
//    auto info = sevenScenes::createInfoVector();
//    functions::createImageVector(all, info, 0);
//    unordered_map<string, cv::Scalar> seq_colors;
//    random_device generator;
//    uniform_int_distribution<int> distribution(1,255);
//    for (int i = 1; i < 15; i++) {
//        Scalar color = Scalar(distribution(generator), distribution(generator), distribution(generator));
//        if (i < 10) {
//            seq_colors["0" + to_string(i)] = color;
//        } else {
//            seq_colors[to_string(i)] = color;
//        }
//    }
//
////    functions::showTop1000(query, 200, 1.6, 20);
//
//    functions::projectCentersTo2D(query, all, seq_colors, query + " All");
//
//    auto top1000 = functions::getTopN(query, 1000);
//    unordered_set<string> unincluded_all (all.begin(), all.end());
//    for (const auto & t : top1000) {
//        unincluded_all.erase(t);
//    }
//    functions::projectCentersTo2D(query, top1000, seq_colors, query + " top 1000");
//
//    auto retrieved = functions::retrieveSimilar(query, 1000, 1.6);
//    unordered_set<string> unincluded_top1000 (top1000.begin(), top1000.end());
//    for (const auto & r : retrieved) {
//        unincluded_top1000.erase(r);
//    }
//    functions::projectCentersTo2D(query, retrieved, seq_colors, query + " Retrieved");
//
//    functions::showAllImages(query, all, seq_colors, unincluded_all, unincluded_top1000, "All Images");
//
//    auto spaced = functions::optimizeSpacing(retrieved, 20, false);
//    functions::projectCentersTo2D(query, spaced, seq_colors, query + " Spaced");




//// Testing Whole Dataset
    vector<string> listQuery;
    vector<tuple<string, string, vector<string>, vector<string>>> info = sevenScenes::createInfoVector();
    functions::createQueryVector(listQuery, info, SCENE);

    cout << "Running queries..." << endl;
    int startIdx = 0;
    vector<double> c_error; c_error.reserve(listQuery.size());
    vector<double> t_calc_error; t_calc_error.reserve(listQuery.size());
    vector<double> r_calc_error; r_calc_error.reserve(listQuery.size());
    vector<double> t_adjust_error; t_adjust_error.reserve(listQuery.size());
    vector<double> r_adjust_error; r_adjust_error.reserve(listQuery.size());
    vector<double> t_avg_error; t_avg_error.reserve(listQuery.size());
    vector<double> r_avg_error; r_avg_error.reserve(listQuery.size());

    vector<double> x (51);
    for (int i = 0; i < 51; i++) {
        x[i] = double (i / 10.);
    }
    vector<double> y_real (51);
    vector<double> y_calc (51);
    vector<double> y_adjust (51);
    vector<double> y_avg (51);
    int total_points = 0;
    for (int q = startIdx; q < listQuery.size(); q++) {

        cout << q + 1 << "/" << listQuery.size() << "  ";

        // Query Image
        string query = listQuery[q];

        // Get query absolute pose and camera intrinsics
        Eigen::Vector3d c_q = sevenScenes::getT(query);
        Eigen::Matrix3d R_q;
        Eigen::Vector3d t_q;
        sevenScenes::getAbsolutePose(query, R_q, t_q);
        double K[4] = {523.538, 529.669, 314.245, 237.595};

        // Get relative poses and filter bad images
        vector<Eigen::Matrix3d> R_ks, R_qk_calcs;
        vector<Eigen::Vector3d> t_ks, t_qk_calcs;

        //Retrieve top K
        auto retrieved = functions::retrieveSimilar(query, 150, 1.6);
        auto spaced = functions::optimizeSpacing(retrieved, 15, false);

        vector<vector<pair<cv::Point2d, cv::Point2d>>> all_GT_inliers; // q, db
        vector<vector<tuple<Point2d, Point2d, double>>> all_points;
        for (const auto & im : spaced) {
            Eigen::Matrix3d R_k, R_qk_calc, R_qk_real;
            Eigen::Vector3d t_k, t_qk_calc, t_qk_real;
            sevenScenes::getAbsolutePose(im, R_k, t_k);

            vector<tuple<Point2d, Point2d, double>> points;
            functions::findMatchesSorted(im, query, "SIFT", 0.8, points);

            if (points.size() < 5) continue;

            vector<cv::Point2d> pts_db, pts_q;
            for (const auto & tup : points) {
                pts_q.push_back(get<0>(tup));
                pts_db.push_back(get<1>(tup));
            }

            if(!functions::getRelativePose(pts_db, pts_q, K, R_qk_calc, t_qk_calc)) continue;

            R_qk_real = R_q * R_k.transpose();
            t_qk_real = t_q - R_qk_real * t_k;
            t_qk_real.normalize();
            Eigen::Matrix3d t_cross_real {
                    {0,               -t_qk_real(2), t_qk_real(1)},
                    {t_qk_real(2),  0,              -t_qk_real(0)},
                    {-t_qk_real(1), t_qk_real(0),               0}};
            Eigen::Matrix3d E_real = t_cross_real * R_qk_real;
            Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
                                  {0.,   K[1], K[3]},
                                  {0.,   0.,   1.}};
            Eigen::Matrix3d F = K_eig.inverse().transpose() * E_real * K_eig.inverse();
            vector<pair<cv::Point2d, cv::Point2d>> GT_inliers = functions::findInliersForFundamental(F, 5., points);

            all_GT_inliers.push_back(GT_inliers);
            all_points.push_back(points);
            R_ks.push_back(R_k);
            t_ks.push_back(t_k);
            R_qk_calcs.push_back(R_qk_calc);
            t_qk_calcs.push_back(t_qk_calc);
        }

        if (R_ks.size() < 2) {
            cout << "Bad query" << endl;
            continue;
        }

        // Calculate query estimates
        vector<Eigen::Matrix3d> R_ks_in = R_ks, R_qk_calcs_in = R_qk_calcs;
        vector<Eigen::Vector3d> t_ks_in = t_ks, t_qk_calcs_in = t_qk_calcs;

        int s = int (R_ks_in.size());
        Eigen::Vector3d c_q_calc = pose::hypothesizeQueryCenterRANSAC(R_ks_in, t_ks_in,R_qk_calcs_in, t_qk_calcs_in);
        int f = int (R_ks_in.size());
        cout << "Inliers: " << f << "/" << s;

        vector<Eigen::Matrix3d> rotations;
        rotations.reserve(R_ks_in.size());
        for(int i = 0; i < R_ks_in.size(); i++) {
            rotations.emplace_back(R_qk_calcs_in[i]*R_ks_in[i]);
        }
        Eigen::Matrix3d R_q_avg = pose::rotationAverage(rotations);

        Eigen::Matrix3d R_q_calc = pose::hypothesizeQueryRotation(c_q_calc,R_q_avg, R_ks_in, t_ks_in, R_qk_calcs_in, t_qk_calcs_in);

        Eigen::Vector3d t_q_calc = -R_q_calc * c_q_calc;
        Eigen::Vector3d t_q_avg = -R_q_avg * c_q_calc;

        Eigen::Vector3d t_q_adjust = t_q_avg;
        Eigen::Matrix3d R_q_adjust = R_q_avg;
        pose::adjustHypothesis(R_ks, t_ks, all_points, K, R_q_adjust, t_q_adjust);


        // Calculate error
        double c_dist = functions::getDistBetween(c_q, c_q_calc);
        c_error.push_back(c_dist); sort(c_error.begin(), c_error.end());

        double t_dist_calc = functions::getDistBetween(t_q, t_q_calc);
        t_calc_error.push_back(t_dist_calc); sort(t_calc_error.begin(), t_calc_error.end());

        double t_dist_avg = functions::getDistBetween(t_q, t_q_avg);
        t_avg_error.push_back(t_dist_avg); sort(t_avg_error.begin(), t_avg_error.end());

        double t_dist_adjust = functions::getDistBetween(t_q, t_q_adjust);
        t_adjust_error.push_back(t_dist_adjust); sort(t_adjust_error.begin(), t_adjust_error.end());

        double r_dist_calc = functions::rotationDifference(R_q, R_q_calc);
        r_calc_error.push_back(r_dist_calc); sort(r_calc_error.begin(), r_calc_error.end());

        double r_dist_avg = functions::rotationDifference(R_q, R_q_avg);
        r_avg_error.push_back(r_dist_avg); sort(r_avg_error.begin(), r_avg_error.end());

        double r_dist_adjust = functions::rotationDifference(R_q, R_q_adjust);
        r_adjust_error.push_back(r_dist_adjust); sort(r_adjust_error.begin(), r_adjust_error.end());

        // Reprojection metrics
        for (int i = 0; i < s; i++) {
            if (all_GT_inliers[i].empty()) continue;

            string im = spaced[i];
            Eigen::Matrix3d R_k = R_ks[i];
            Eigen::Vector3d t_k = t_ks[i];

            vector<cv::Point2d> pts_db, pts_q;
            for (const auto & p : all_GT_inliers[i]) {
                pts_q.push_back(p.first);
                pts_db.push_back(p.second);
            }
            total_points += int (pts_db.size());

            //// Real
            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
            Eigen::Vector3d t_qk_real = t_q - R_qk_real * t_k;

            Eigen::Matrix3d t_cross_real {
                    {0,               -t_qk_real(2), t_qk_real(1)},
                    {t_qk_real(2),  0,              -t_qk_real(0)},
                    {-t_qk_real(1), t_qk_real(0),               0}};
            Eigen::Matrix3d E_real = t_cross_real * R_qk_real;

            Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
                                  {0.,   K[1], K[3]},
                                  {0.,   0.,   1.}};

            Eigen::Matrix3d F_real_eig = K_eig.inverse().transpose() * E_real * K_eig.inverse();
            cv::Mat F_real; cv::eigen2cv(F_real_eig, F_real);

            vector<cv::Point3d> lines_real;
            cv::computeCorrespondEpilines(pts_db, 1, F_real, lines_real);

            for (int p = 0; p < lines_real.size(); p++) {
                auto l = lines_real[p];
                auto pt = pts_q[p];

                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
                int idx = int (round(error*10));
                if (idx <= 50) {
                    y_real[idx]++;
                }
            }

            //// Calc
            Eigen::Matrix3d R_qk_calc = R_q_avg * R_k.transpose();
            Eigen::Vector3d t_qk_calc = t_q_calc - R_qk_calc * t_k;
            t_qk_calc.normalize();

            Eigen::Matrix3d t_cross_calc {
                    {0,               -t_qk_calc(2), t_qk_calc(1)},
                    {t_qk_calc(2),  0,              -t_qk_calc(0)},
                    {-t_qk_calc(1), t_qk_calc(0),               0}};
            Eigen::Matrix3d E_calc = t_cross_calc * R_qk_calc;

            Eigen::Matrix3d F_calc_eig = K_eig.inverse().transpose() * E_calc * K_eig.inverse();
            cv::Mat F_calc; cv::eigen2cv(F_calc_eig, F_calc);

            vector<cv::Point3d> lines_calc;
            cv::computeCorrespondEpilines(pts_db, 1, F_calc, lines_calc);

            for (int p = 0; p < lines_calc.size(); p++) {
                auto l = lines_calc[p];
                auto pt = pts_q[p];

                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
                int idx = int (round(error*10));
                if (idx <= 50) {
                    y_calc[idx]++;
                }
            }

            //// Adjust
            Eigen::Matrix3d R_qk_adjust = R_q_adjust * R_k.transpose();
            Eigen::Vector3d t_qk_adjust = t_q_adjust - R_qk_adjust * t_k;
            t_qk_adjust.normalize();

            Eigen::Matrix3d t_cross_adjust {
                    {0,               -t_qk_adjust(2), t_qk_adjust(1)},
                    {t_qk_adjust(2),  0,              -t_qk_adjust(0)},
                    {-t_qk_adjust(1), t_qk_adjust(0),               0}};
            Eigen::Matrix3d E_adjust = t_cross_adjust * R_qk_adjust;

            Eigen::Matrix3d F_adjust_eig = K_eig.inverse().transpose() * E_adjust * K_eig.inverse();
            cv::Mat F_adjust; cv::eigen2cv(F_adjust_eig, F_adjust);

            vector<cv::Point3d> lines_adjust;
            cv::computeCorrespondEpilines(pts_db, 1, F_adjust, lines_adjust);

            for (int p = 0; p < lines_adjust.size(); p++) {
                auto l = lines_adjust[p];
                auto pt = pts_q[p];

                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
                int idx = int (round(error*10));
                if (idx <= 50) {
                    y_adjust[idx]++;
                }
            }

            //// Avg
            Eigen::Matrix3d R_qk_avg = R_q_avg * R_k.transpose();
            Eigen::Vector3d t_qk_avg = t_q_avg - R_qk_avg * t_k;
            t_qk_avg.normalize();

            Eigen::Matrix3d t_cross_avg {
                    {0,               -t_qk_avg(2), t_qk_avg(1)},
                    {t_qk_avg(2),  0,              -t_qk_avg(0)},
                    {-t_qk_avg(1), t_qk_avg(0),               0}};
            Eigen::Matrix3d E_avg = t_cross_avg * R_qk_avg;

            Eigen::Matrix3d F_avg_eig = K_eig.inverse().transpose() * E_avg * K_eig.inverse();
            cv::Mat F_avg; cv::eigen2cv(F_avg_eig, F_avg);

            vector<cv::Point3d> lines_avg;
            cv::computeCorrespondEpilines(pts_db, 1, F_avg, lines_avg);

            for (int p = 0; p < lines_avg.size(); p++) {
                auto l = lines_avg[p];
                auto pt = pts_q[p];

                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
                int idx = int (round(error*10));
                if (idx <= 50) {
                    y_avg[idx]++;
                }
            }
        }

        int n = int(double(c_error.size()) / 2.);
        if (c_error.size() % 2 == 1) {
            cout <<
            ", C: " << c_error[n] <<
            ", T_calc: " << t_calc_error[n] <<
            ", R_calc: " << r_calc_error[n] <<
            ", T_adjust: " << t_adjust_error[n] <<
            ", R_adjust: " << r_adjust_error[n] <<
            ", T_avg: " << t_avg_error[n] <<
            ", R_avg: " << r_avg_error[n] <<
            endl;
        } else {
            cout <<
            ", C: " << (c_error[n] + c_error[n - 1]) / 2. <<
            ", T_calc: " << (t_calc_error[n] + t_calc_error[n - 1]) / 2. <<
            ", R_calc: " << (r_calc_error[n] + r_calc_error[n - 1]) / 2. <<
            ", T_adjust: " << (t_adjust_error[n] + t_adjust_error[n - 1]) / 2. <<
            ", R_adjust: " << (r_adjust_error[n] + r_adjust_error[n - 1]) / 2.<<
            ", T_avg: " << (t_avg_error[n] + t_avg_error[n - 1]) / 2. <<
            ", R_avg: " << (r_avg_error[n] + r_avg_error[n - 1]) / 2. <<
            endl;
        }
    }
    for (int i = 0; i < y_real.size(); i++) {
        y_real[i] *= 100./total_points;
        y_calc[i] *= 100./total_points;
        y_adjust[i] *= 100./total_points;
        y_avg[i] *= 100./total_points;
    }


    plt::figure_size(1000, 800);
    plt::plot(x, y_real);
    plt::xlim(0, 5);
    plt::xlabel("Reprojection Error (px)");
    plt::ylim(0, 15);
    plt::ylabel("Percentage of Points");
    plt::title("Real Reprojection");
    plt::show();

    plt::figure_size(1000, 800);
    plt::plot(x, y_calc);
    plt::xlim(0, 10);
    plt::xlabel("Reprojection Error (px)");
    plt::ylim(0, 15);
    plt::ylabel("Percentage of Points");
    plt::title("Calc Reprojection");
    plt::show();

    plt::figure_size(1000, 800);
    plt::plot(x, y_adjust);
    plt::xlim(0, 5);
    plt::xlabel("Reprojection Error (px)");
    plt::ylim(0, 15);
    plt::ylabel("Percentage of Points");
    plt::title("Adjust Reprojection");
    plt::show();

    plt::figure_size(1000, 800);
    plt::plot(x, y_avg);
    plt::xlim(0, 5);
    plt::xlabel("Reprojection Error (px)");
    plt::ylim(0, 15);
    plt::ylabel("Percentage of Points");
    plt::title("Avg Reprojection");
    plt::show();

    return 0;
}