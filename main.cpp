
#include "include/sevenScenes.h"
#include "include/synthetic.h"
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
#include "include/OptimalRotationSolver.h"



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

//    vector<Eigen::Matrix3d> R_k;
//    vector<Eigen::Vector3d> t_k;
//    vector<Eigen::Matrix3d> R_qk;
//    vector<Eigen::Vector3d> t_qk;
//
//    Eigen::Matrix3d R_q {{0.93118482425723736462, 0.36262296659846848801, -0.037408650982680001496},
//                           {0.33249792459696492219, -0.8027621803172360071, 0.49499294135475135903},
//                           {0.14946555861377897045, -0.47336821391771044532, -0.86809134360424744514}};
//    Eigen::Vector3d c_q {-157.37370535944677385, 551.2138747105557286, 985.10057207662759993};
//    Eigen::Vector3d t_q = -R_q * c_q;
//
//
//    Eigen::Matrix3d R_1 {{0.55072336929004850337, -0.82712030047107198971, 0.11214178109188482901},
//                         {-0.11722689697755828142, 0.056375599209274707135, 0.99150372991674018408},
//                         {-0.8264149231123731898, -0.5591903078223356971, -0.065913386309092270032}};
//    Eigen::Vector3d c_1 {938.1809157763350413, 613.74851039172017408, 55.011595897134093036};
//    Eigen::Vector3d t_1 = -R_1 * c_1;
//    Eigen::Matrix3d R_q1 = R_q * R_1.transpose();
//    Eigen::Vector3d t_q1 = t_q - R_q1 * t_1;
//    t_q1.normalize();
//    R_k.push_back(R_1);
//    t_k.push_back(t_1);
//    R_qk.push_back(R_q1);
//    t_qk.push_back(t_q1);
//
//    Eigen::Matrix3d R_2 {{0.088461304994031042526, 0.033230569322564879053, 0.99552515125497997861},
//                         {-0.22987060353240071353, -0.9717843825228138499, 0.052864160982681845935},
//                         {0.9691925005644660418, -0.2335184000186093789, -0.078326583624883427959}};
//    Eigen::Vector3d c_2 {-1091.5951653892184368, 272.98698735737838206, 107.90863343431843191};
//    Eigen::Vector3d t_2 = -R_2 * c_2;
//    Eigen::Matrix3d R_q2 = R_q * R_2.transpose();
//    Eigen::Vector3d t_q2 = t_q - R_q2 * t_2;
//    t_q2.normalize();
//    R_k.push_back(R_2);
//    t_k.push_back(t_2);
//    R_qk.push_back(R_q2);
//    t_qk.push_back(t_q2);
//
//    Eigen::Matrix3d R_3 {{-0.10387879175474701299, 0.8104268925167098514, 0.5765565440694283561},
//                         {-0.9498195017811733587, 0.091132400555301251721, -0.29922867443677803045},
//                         {-0.29504594669260764128, -0.57870816259556667749, 0.76029254362089171426}};
//    Eigen::Vector3d c_3 {346.58158737322162324, 638.29359688719728183, -848.51081310189067608};
//    Eigen::Vector3d t_3 = -R_3 * c_3;
//    Eigen::Matrix3d R_q3 = R_q * R_3.transpose();
//    Eigen::Vector3d t_q3 = t_q - R_q3 * t_3;
//    t_q3.normalize();
//    R_k.push_back(R_3);
//    t_k.push_back(t_3);
//    R_qk.push_back(R_q3);
//    t_qk.push_back(t_q3);
//
//    auto start = high_resolution_clock::now();
//    Eigen::Matrix3d R_hyp = rotation::solve_rotation(c_q, R_k, t_k, R_qk, t_qk);
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<microseconds>(stop - start);
//    cout << duration.count() << "microseconds"  << endl;
//    double r_dist = functions::rotationDifference(R_q, R_hyp);
//    cout << r_dist << endl;
//
//    int done = 0;






//
//    vector<string> images = synthetic::getAll();
//    vector<Point2d> pts_0, pts_1;
//
//    synthetic::findSyntheticMatches(images[0], images[1], pts_0, pts_1);
//
//    synthetic::addMismatches(pts_0, pts_1, .10);
//
//
//
//    vector<Eigen::Vector3d> points3d = synthetic::get3DPoints();
//    Eigen::Matrix3d K_eig{{2584.9325098195013197, 0.,                    249.77137587221417903},
//                          {0.,                    2584.7918606057692159, 278.31267937919352562},
//                          {0.,                    0.,                                       1.}};
//    Eigen::Matrix3d R;
//    Eigen::Vector3d T;
//    synthetic::getAbsolutePose(images[0], R, T);
//    for(int i = 0; i < points3d.size(); i++) {
//        Eigen::Vector3d point2d {pts_0[i].x, pts_0[i].y, 1.};
//        Eigen::Vector3d proj2d = R * points3d[i] + T;
//        proj2d = proj2d / proj2d[2];
//        proj2d = K_eig * proj2d;
//
//        double dist = sqrt(pow((point2d[0] - proj2d[0]), 2.) + pow((point2d[1] - proj2d[1]), 2.));
//
//        cout << dist << endl;
//
//
//    }
//
//
//    exit(0);

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
//    functions::showTop1000(query, 100, 1.6, 10);
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
    vector<string> listQuery = synthetic::getAll();
//    vector<tuple<string, string, vector<string>, vector<string>>> info = sevenScenes::createInfoVector();
//    functions::createQueryVector(listQuery, info, SCENE);

    cout << "Running queries..." << endl;
    int startIdx = 0;
    vector<double> t_alone_error; t_alone_error.reserve(listQuery.size());
    vector<double> r_alone_error; r_alone_error.reserve(listQuery.size());
    vector<double> t_alone_ad_error; t_alone_ad_error.reserve(listQuery.size());
    vector<double> r_alone_ad_error; r_alone_ad_error.reserve(listQuery.size());

    vector<double> t_avg_error; t_avg_error.reserve(listQuery.size());
    vector<double> r_avg_error; r_avg_error.reserve(listQuery.size());
    vector<double> t_avg_ad_error; t_avg_ad_error.reserve(listQuery.size());
    vector<double> r_avg_ad_error; r_avg_ad_error.reserve(listQuery.size());

    vector<double> t_gov_error; t_gov_error.reserve(listQuery.size());
    vector<double> r_gov_error; r_gov_error.reserve(listQuery.size());
    vector<double> t_gov_ad_error; t_gov_ad_error.reserve(listQuery.size());
    vector<double> r_gov_ad_error; r_gov_ad_error.reserve(listQuery.size());

    vector<double> t_calc_error; t_calc_error.reserve(listQuery.size());
    vector<double> r_calc_error; r_calc_error.reserve(listQuery.size());
    vector<double> t_calc_ad_error; t_calc_ad_error.reserve(listQuery.size());
    vector<double> r_calc_ad_error; r_calc_ad_error.reserve(listQuery.size());

    vector<double> x (51);
    for (int i = 0; i < 51; i++) {
        x[i] = double (i / 10.);
    }
    vector<double> t (151);
    for (int i = 0; i < 151; i++) {
        t[i] = double (i / 10.);
    }
    vector<double> tpr_vec;
    vector<double> fpr_vec;

    vector<double> y_real (51);
    vector<double> y_calc (51);
    vector<double> y_adjust (51);
    vector<double> y_avg (51);
    int total_points = 0;
//    ofstream distribution;
//    distribution.open ("/Users/cameronfiore/C++/image_localization_project/data/distribution.txt");
    for (int q = startIdx; q < listQuery.size(); q++) {

        cout << q + 1 << "/" << listQuery.size() << ": ";

        // Query Image
        string query = listQuery[q];

        // Get query absolute pose and camera intrinsics
        Eigen::Vector3d c_q = synthetic::getC(query);
//        Eigen::Vector3d c_q = sevenScenes::getT(query);
        Eigen::Matrix3d R_q;
        Eigen::Vector3d t_q;
//        sevenScenes::getAbsolutePose(query, R_q, t_q);
        synthetic::getAbsolutePose(query, R_q, t_q);

//        double K[4] = {523.538, 529.669, 314.245, 237.595};
        double K[4] = {2584.9325098195013197, 2584.7918606057692159, 249.77137587221417903, 278.31267937919352562};


        //Retrieve top K
        auto retrieved = synthetic::omitQuery(q, listQuery);
//        auto retrieved = functions::retrieveSimilar(query, 150, 1.35);
//        int num_spaced = int(double(retrieved.size()) * .1);
//        auto spaced = functions::optimizeSpacing(retrieved, 15, false, "7-Scenes");
        auto spaced = functions::optimizeSpacing(retrieved, 10, false, "synthetic");



//        for (int i = 0; i < spaced.size() - 1; i++) {
//            for (int j = i + 1; j < spaced.size(); j++) {
//                Eigen::Matrix3d R_i = sevenScenes::getR(spaced[i]);
//                Eigen::Matrix3d R_j = sevenScenes::getR(spaced[j]);
//
//                Eigen::Vector3d z {0, 0, 1};
//
//                Eigen::Vector3d dir_i = R_i * z;
//                Eigen::Vector3d dir_j = R_j * z;
//
//                double dist = functions::getAngleBetween(dir_i, dir_j);
//
//                distribution << dist << endl;
//
//            }
//        }
//         Get relative poses and filter bad images
        vector<Eigen::Matrix3d> R_ks, R_qk_calcs;
        vector<Eigen::Vector3d> t_ks, t_qk_calcs;
        vector<int> inliers;
        vector<vector<pair<cv::Point2d, cv::Point2d>>> all_GT_inliers; // q, db
        vector<vector<tuple<Point2d, Point2d, double>>> all_points;
        vector<string> anchors;
        for (const auto & im : spaced) {
            Eigen::Matrix3d R_k, R_qk_calc, R_qk_real;
            Eigen::Vector3d t_k, t_qk_calc, t_qk_real;
//            sevenScenes::getAbsolutePose(im, R_k, t_k);
            synthetic::getAbsolutePose(im, R_k, t_k);


//            vector<tuple<Point2d, Point2d, double>> points;
//            functions::findMatchesSorted(im, query, "SIFT", 0.8, points);
//
//            if (points.size() < 5) continue;

            vector<cv::Point2d> pts_db, pts_q;
            synthetic::findSyntheticMatches(im, query, pts_db, pts_q);


            synthetic::addGaussianNoise(pts_db, pts_q, .1);
            synthetic::addOcclusion(pts_db, pts_q, 1000);
            synthetic::addMismatches(pts_db, pts_q, .10);


            vector<tuple<Point2d, Point2d, double>> points;
            points.reserve(pts_db.size());
            for(int i = 0; i < pts_db.size(); i++) {
                points.emplace_back(pts_q[i], pts_db[i], 0.);
            }
//            for (const auto & tup : points) {
//                pts_q.push_back(get<0>(tup));
//                pts_db.push_back(get<1>(tup));
//            }

            if(!functions::getRelativePose(pts_db, pts_q, K, R_qk_calc, t_qk_calc)) continue;

            R_qk_real = R_q * R_k.transpose();
            t_qk_real = t_q - R_qk_real * t_k;
            t_qk_real.normalize();

            double r_dist = functions::rotationDifference(R_qk_real, R_qk_calc);
            double t_dist = functions::getAngleBetween(t_qk_real, t_qk_calc);

            if (r_dist <= 10. && t_dist <= 10.) {
                inliers.push_back(1);
            } else {
                inliers.push_back(0);
            }

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

            anchors.push_back(im);
            all_GT_inliers.push_back(GT_inliers);
            all_points.push_back(points);
            R_ks.push_back(R_k);
            t_ks.push_back(t_k);
            R_qk_calcs.push_back(R_qk_calc);
            t_qk_calcs.push_back(t_qk_calc);
        }

        if (anchors.size() < 2) {
            cout << "Bad query" << endl;
            continue;
        }

//        anchors.push_back(query);
//        auto BA_result = pose::bundleAdjust(K, anchors);
//        for(int i = 0; i < anchors.size() - 1; i++) {
//            double angle_change = functions::rotationDifference(R_ks[i], BA_result.first[i]);
//            double dist_change = functions::getDistBetween(t_ks[i], BA_result.second[i]);
//            int stop = 0;
//        }
//        double query_angle_change = functions::rotationDifference(R_q, BA_result.first[anchors.size()-1]);
//        double query_dist_change = functions::getDistBetween(t_q, BA_result.second[anchors.size()-1]);
//        R_ks.clear();
//        t_ks.clear();
//        for (int i = 0; i < anchors.size() - 1; i++) {
//            R_ks.push_back(BA_result.first[i]);
//            t_ks.push_back(BA_result.second[i]);
//        }
//        R_q = BA_result.first[anchors.size()-1];
//        t_q = BA_result.second[anchors.size()-1];
//        c_q = -R_q.transpose() * t_q;

//
//        int num_inliers = 0;
//        for(const auto & i : inliers) {
//            if (i) {
//                num_inliers++;
//            }
//        }
//        int num_outliers = int (inliers.size()) - num_inliers;
//        for(const auto & t_val : t) {
//
//            auto calc = pose::hypothesizeRANSAC(t_val, t_val, inliers, R_ks, t_ks, R_qk_calcs, t_qk_calcs, c_q, R_q);\
//            tpr_vec.push_back(get<4>(calc) / num_inliers);
//            fpr_vec.push_back(get<5>(calc) / num_outliers);
//        }
//
//        plt::figure_size(1000, 800);
//        plt::plot(fpr_vec, tpr_vec);
//        plt::xlim(-.1, 1.1);
//        plt::xlabel("False Positive Rate");
//        plt::ylim(-.1, 1.1);
//        plt::ylabel("True Positive Rate");
//        plt::title("ROC Plot");
//        plt::show();
//
//        plt::figure_size(1000, 800);
//        plt::plot(t, fpr_vec);
//        plt::xlim(0., 15.);
//        plt::xlabel("Threshold Value");
//        plt::ylim(0., 1.);
//        plt::ylabel("False Positive Rate");
//        plt::title("Threshold Plot");
//        plt::show();
//
//        plt::figure_size(1000, 800);
//        plt::plot(t, tpr_vec);
//        plt::xlim(0., 15.);
//        plt::xlabel("Threshold Value");
//        plt::ylim(0., 1.);
//        plt::ylabel("True Positive Rate");
//        plt::title("Threshold Plot");
//        plt::show();

        // Calculate query estimates

//        for (int i = 0; i < 25; i++) {
//
//            auto calc = pose::hypothesizeRANSAC(5., 5., inliers, R_ks, t_ks, R_qk_calcs, t_qk_calcs, c_q, R_q);
//            int in = get<7>(calc);
//            Eigen::Matrix3d R_q_avg = get<3>(calc);
//            Eigen::Matrix3d R_q_calc = get<5>(calc);
//
//            double r_avg_dist = functions::rotationDifference(R_q, R_q_avg);
//            double r_calc_dist = functions::rotationDifference(R_q, R_q_calc);
//
//            int check = 0;
//
//        }


        auto calc = pose::hypothesizeRANSAC(5., 5., inliers, R_ks, t_ks, R_qk_calcs, t_qk_calcs, c_q, R_q);
        cout << "Inliers: " << get<7>(calc) << "/" << R_ks.size() << endl;

//        vector<Eigen::Matrix3d> rotations;
//        rotations.reserve(R_ks.size());
//        for(int i = 0; i < R_ks.size(); i++) {
//            rotations.emplace_back(R_qk_calcs[i] * R_ks[i]);
//        }
//        Eigen::Matrix3d R_q_avg = pose::rotationAverage(rotations);

        Eigen::Vector3d t_q_alone = get<0>(calc);
        Eigen::Matrix3d R_q_alone = get<1>(calc);
        Eigen::Vector3d t_q_alone_ad = t_q_alone;
        Eigen::Matrix3d R_q_alone_ad = R_q_alone;
//        pose::adjustHypothesis(R_ks, t_ks, all_points, 10., K, R_q_alone_ad, t_q_alone_ad);

        Eigen::Vector3d t_q_avg = get<2>(calc);
        Eigen::Matrix3d R_q_avg = get<3>(calc);
        Eigen::Vector3d t_q_avg_ad = t_q_avg;
        Eigen::Matrix3d R_q_avg_ad = R_q_avg;
//        pose::adjustHypothesis(R_ks, t_ks, all_points, 10., K, R_q_avg_ad, t_q_avg_ad);

        Eigen::Vector3d t_q_gov = get<6>(calc);
        Eigen::Matrix3d R_q_gov = get<3>(calc);
        Eigen::Vector3d t_q_gov_ad = t_q_gov;
        Eigen::Matrix3d R_q_gov_ad = R_q_gov;
//        pose::adjustHypothesis(R_ks, t_ks, all_points, 10., K, R_q_gov_ad, t_q_gov_ad);

        Eigen::Vector3d t_q_calc = get<4>(calc);
        Eigen::Matrix3d R_q_calc = get<5>(calc);
        Eigen::Vector3d t_q_calc_ad = t_q_calc;
        Eigen::Matrix3d R_q_calc_ad = R_q_calc;
//        pose::adjustHypothesis(R_ks, t_ks, all_points, 10., K, R_q_calc_ad, t_q_calc_ad);

        // Calculate error

        double t_alone_dist = functions::getDistBetween(t_q, t_q_alone);
        t_alone_error.push_back(t_alone_dist); sort(t_alone_error.begin(), t_alone_error.end());
        double r_alone_dist = functions::rotationDifference(R_q, R_q_alone);
        r_alone_error.push_back(r_alone_dist); sort(r_alone_error.begin(), r_alone_error.end());
        double t_alone_ad_dist = functions::getDistBetween(t_q, t_q_alone_ad);
        t_alone_ad_error.push_back(t_alone_ad_dist); sort(t_alone_ad_error.begin(), t_alone_ad_error.end());
        double r_alone_ad_dist = functions::rotationDifference(R_q, R_q_alone_ad);
        r_alone_ad_error.push_back(r_alone_ad_dist); sort(r_alone_ad_error.begin(), r_alone_ad_error.end());

        double t_avg_dist = functions::getDistBetween(t_q, t_q_avg);
        t_avg_error.push_back(t_avg_dist); sort(t_avg_error.begin(), t_avg_error.end());
        double r_avg_dist = functions::rotationDifference(R_q, R_q_avg);
        r_avg_error.push_back(r_avg_dist); sort(r_avg_error.begin(), r_avg_error.end());
        double t_avg_ad_dist = functions::getDistBetween(t_q, t_q_avg_ad);
        t_avg_ad_error.push_back(t_avg_ad_dist); sort(t_avg_ad_error.begin(), t_avg_ad_error.end());
        double r_avg_ad_dist = functions::rotationDifference(R_q, R_q_avg_ad);
        r_avg_ad_error.push_back(r_avg_ad_dist); sort(r_avg_ad_error.begin(), r_avg_ad_error.end());

        double t_gov_dist = functions::getDistBetween(t_q, t_q_gov);
        t_gov_error.push_back(t_gov_dist); sort(t_gov_error.begin(), t_gov_error.end());
        double r_gov_dist = functions::rotationDifference(R_q, R_q_gov);
        r_gov_error.push_back(r_gov_dist); sort(r_gov_error.begin(), r_gov_error.end());
        double t_gov_ad_dist = functions::getDistBetween(t_q, t_q_gov_ad);
        t_gov_ad_error.push_back(t_gov_ad_dist); sort(t_gov_ad_error.begin(), t_gov_ad_error.end());
        double r_gov_ad_dist = functions::rotationDifference(R_q, R_q_gov_ad);
        r_gov_ad_error.push_back(r_gov_ad_dist); sort(r_gov_ad_error.begin(), r_gov_ad_error.end());

        double t_calc_dist = functions::getDistBetween(t_q, t_q_calc);
        t_calc_error.push_back(t_calc_dist); sort(t_calc_error.begin(), t_calc_error.end());
        double r_calc_dist = functions::rotationDifference(R_q, R_q_calc);
        r_calc_error.push_back(r_calc_dist); sort(r_calc_error.begin(), r_calc_error.end());
        double t_calc_ad_dist = functions::getDistBetween(t_q, t_q_calc_ad);
        t_calc_ad_error.push_back(t_calc_ad_dist); sort(t_calc_ad_error.begin(), t_calc_ad_error.end());
        double r_calc_ad_dist = functions::rotationDifference(R_q, R_q_calc_ad);
        r_calc_ad_error.push_back(r_calc_ad_dist); sort(r_calc_ad_error.begin(), r_calc_ad_error.end());



        // Reprojection metrics
//        for (int i = 0; i < int (R_ks.size()); i++) {
//            if (all_GT_inliers[i].empty()) continue;
//
//            string im = spaced[i];
//            Eigen::Matrix3d R_k = R_ks[i];
//            Eigen::Vector3d t_k = t_ks[i];
//
//            vector<cv::Point2d> pts_db, pts_q;
//            for (const auto & p : all_GT_inliers[i]) {
//                pts_q.push_back(p.first);
//                pts_db.push_back(p.second);
//            }
//            total_points += int (pts_db.size());
//
//            //// Real
//            Eigen::Matrix3d R_qk_real = R_q * R_k.transpose();
//            Eigen::Vector3d t_qk_real = t_q - R_qk_real * t_k;
//
//            Eigen::Matrix3d t_cross_real {
//                    {0,               -t_qk_real(2), t_qk_real(1)},
//                    {t_qk_real(2),  0,              -t_qk_real(0)},
//                    {-t_qk_real(1), t_qk_real(0),               0}};
//            Eigen::Matrix3d E_real = t_cross_real * R_qk_real;
//
//            Eigen::Matrix3d K_eig{{K[0], 0.,   K[2]},
//                                  {0.,   K[1], K[3]},
//                                  {0.,   0.,   1.}};
//
//            Eigen::Matrix3d F_real_eig = K_eig.inverse().transpose() * E_real * K_eig.inverse();
//            cv::Mat F_real; cv::eigen2cv(F_real_eig, F_real);
//
//            vector<cv::Point3d> lines_real;
//            cv::computeCorrespondEpilines(pts_db, 1, F_real, lines_real);
//
//            for (int p = 0; p < lines_real.size(); p++) {
//                auto l = lines_real[p];
//                auto pt = pts_q[p];
//
//                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
//                int idx = int (round(error*10));
//                if (idx <= 50) {
//                    y_real[idx]++;
//                }
//            }
//
//            //// Calc
//            Eigen::Matrix3d R_qk_calc = R_q_avg * R_k.transpose();
//            Eigen::Vector3d t_qk_calc = t_q_calc - R_qk_calc * t_k;
//            t_qk_calc.normalize();
//
//            Eigen::Matrix3d t_cross_calc {
//                    {0,               -t_qk_calc(2), t_qk_calc(1)},
//                    {t_qk_calc(2),  0,              -t_qk_calc(0)},
//                    {-t_qk_calc(1), t_qk_calc(0),               0}};
//            Eigen::Matrix3d E_calc = t_cross_calc * R_qk_calc;
//
//            Eigen::Matrix3d F_calc_eig = K_eig.inverse().transpose() * E_calc * K_eig.inverse();
//            cv::Mat F_calc; cv::eigen2cv(F_calc_eig, F_calc);
//
//            vector<cv::Point3d> lines_calc;
//            cv::computeCorrespondEpilines(pts_db, 1, F_calc, lines_calc);
//
//            for (int p = 0; p < lines_calc.size(); p++) {
//                auto l = lines_calc[p];
//                auto pt = pts_q[p];
//
//                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
//                int idx = int (round(error*10));
//                if (idx <= 50) {
//                    y_calc[idx]++;
//                }
//            }
//
//            //// Adjust
//            Eigen::Matrix3d R_qk_adjust = R_q_adjust * R_k.transpose();
//            Eigen::Vector3d t_qk_adjust = t_q_adjust - R_qk_adjust * t_k;
//            t_qk_adjust.normalize();
//
//            Eigen::Matrix3d t_cross_adjust {
//                    {0,               -t_qk_adjust(2), t_qk_adjust(1)},
//                    {t_qk_adjust(2),  0,              -t_qk_adjust(0)},
//                    {-t_qk_adjust(1), t_qk_adjust(0),               0}};
//            Eigen::Matrix3d E_adjust = t_cross_adjust * R_qk_adjust;
//
//            Eigen::Matrix3d F_adjust_eig = K_eig.inverse().transpose() * E_adjust * K_eig.inverse();
//            cv::Mat F_adjust; cv::eigen2cv(F_adjust_eig, F_adjust);
//
//            vector<cv::Point3d> lines_adjust;
//            cv::computeCorrespondEpilines(pts_db, 1, F_adjust, lines_adjust);
//
//            for (int p = 0; p < lines_adjust.size(); p++) {
//                auto l = lines_adjust[p];
//                auto pt = pts_q[p];
//
//                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
//                int idx = int (round(error*10));
//                if (idx <= 50) {
//                    y_adjust[idx]++;
//                }
//            }
//
//            //// Avg
//            Eigen::Matrix3d R_qk_avg = R_q_avg * R_k.transpose();
//            Eigen::Vector3d t_qk_avg = t_q_avg - R_qk_avg * t_k;
//            t_qk_avg.normalize();
//
//            Eigen::Matrix3d t_cross_avg {
//                    {0,               -t_qk_avg(2), t_qk_avg(1)},
//                    {t_qk_avg(2),  0,              -t_qk_avg(0)},
//                    {-t_qk_avg(1), t_qk_avg(0),               0}};
//            Eigen::Matrix3d E_avg = t_cross_avg * R_qk_avg;
//
//            Eigen::Matrix3d F_avg_eig = K_eig.inverse().transpose() * E_avg * K_eig.inverse();
//            cv::Mat F_avg; cv::eigen2cv(F_avg_eig, F_avg);
//
//            vector<cv::Point3d> lines_avg;
//            cv::computeCorrespondEpilines(pts_db, 1, F_avg, lines_avg);
//
//            for (int p = 0; p < lines_avg.size(); p++) {
//                auto l = lines_avg[p];
//                auto pt = pts_q[p];
//
//                double error = abs(l.x*pt.x + l.y*pt.y + l.z) / sqrt(l.x*l.x + l.y*l.y);
//                int idx = int (round(error*10));
//                if (idx <= 50) {
//                    y_avg[idx]++;
//                }
//            }
//        }

//        cout <<
//        ", T_calc: " << t_dist_calc <<
//        ", R_calc: " << r_dist_calc <<
//        ", R_avg: " << r_dist_avg <<
//        endl;

        int n = int(double(t_gov_error.size()) / 2.);
        if (t_gov_error.size() % 2 == 1) {
            cout <<
            "T_alone: " << t_alone_error[n] <<
            ", R_alone: " << r_alone_error[n] <<
            ", T_alone_ad: " << t_alone_ad_error[n] <<
            ", R_alone_ad: " << r_alone_ad_error[n] << endl <<

            "T_avg: " << t_avg_error[n] <<
            ", R_avg: " << r_avg_error[n] <<
            ", T_avg_ad: " << t_avg_ad_error[n] <<
            ", R_avg_ad: " << r_avg_ad_error[n] << endl <<

            "T_gov: " << t_gov_error[n] <<
            ", R_gov: " << r_gov_error[n] <<
            ", T_gov_ad: " << t_gov_ad_error[n] <<
            ", R_gov_ad: " << r_gov_ad_error[n] << endl <<

            "T_calc: " << t_calc_error[n] <<
            ", R_calc: " << r_calc_error[n] <<
            ", T_calc_ad: " << t_calc_ad_error[n] <<
            ", R_calc_ad: " << r_calc_ad_error[n] <<
            endl;
        } else {
            cout <<
            "T_alone: " << (t_alone_error[n] + t_alone_error[n - 1]) / 2. <<
            ", R_alone: " << (r_alone_error[n] + r_alone_error[n - 1]) / 2. <<
            ", T_alone_ad: " << (t_alone_ad_error[n] + t_alone_ad_error[n - 1]) / 2. <<
            ", R_alone_ad: " << (r_alone_ad_error[n] + r_alone_ad_error[n - 1]) / 2. << endl <<

            "T_avg: " << (t_avg_error[n] + t_avg_error[n - 1]) / 2. <<
            ", R_avg: " << (r_avg_error[n] + r_avg_error[n - 1]) / 2. <<
            ", T_avg_ad: " << (t_avg_ad_error[n] + t_avg_ad_error[n - 1]) / 2. <<
            ", R_avg_ad: " << (r_avg_ad_error[n] + r_avg_ad_error[n - 1]) / 2. << endl <<

            "T_gov: " << (t_gov_error[n] + t_gov_error[n - 1]) / 2. <<
            ", R_gov: " << (r_gov_error[n] + r_gov_error[n - 1]) / 2. <<
            ", T_gov_ad: " << (t_gov_ad_error[n] + t_gov_ad_error[n - 1]) / 2. <<
            ", R_gov_ad: " << (r_gov_ad_error[n] + r_gov_ad_error[n - 1]) / 2. << endl <<

            "T_calc: " << (t_calc_error[n] + t_calc_error[n - 1]) / 2. <<
            ", R_calc: " << (r_calc_error[n] + r_calc_error[n - 1]) / 2. <<
            ", T_calc_ad: " << (t_calc_ad_error[n] + t_calc_ad_error[n - 1]) / 2. <<
            ", R_calc_ad: " << (r_calc_ad_error[n] + r_calc_ad_error[n - 1]) / 2. <<
            endl;
        }
    }
//    distribution.close();
//    ofstream alone, avg, gov, calc;
//    alone.open ("/Users/cameronfiore/C++/image_localization_project/data/redkitchen/alone.txt");
//    avg.open ("/Users/cameronfiore/C++/image_localization_project/data/redkitchen/avg.txt");
//    gov.open ("/Users/cameronfiore/C++/image_localization_project/data/redkitchen/gov.txt");
//    calc.open ("/Users/cameronfiore/C++/image_localization_project/data/redkitchen/calc.txt");
//    for (int i = 0; i < t_alone_error.size(); i++) {
//        alone << t_alone_error[i] << "   " << r_alone_error[i] << "   " << t_alone_ad_error[i] << "   " << r_alone_ad_error[i] << endl;
//        avg << t_avg_error[i] << "   " << r_avg_error[i] << "   " << t_avg_ad_error[i] << "   " << r_avg_ad_error[i] << endl;
//        gov << t_gov_error[i] << "   " << r_gov_error[i] << "   " << t_gov_ad_error[i] << "   " << r_gov_ad_error[i] << endl;
//        calc << t_calc_error[i] << "   " << r_calc_error[i] << "   " << t_calc_ad_error[i] << "   " << r_calc_ad_error[i] << endl;
//    }
//    alone.close();
//    avg.close();
//    gov.close();
//    calc.close();


//    for (int i = 0; i < y_real.size(); i++) {
//        y_real[i] *= 100./total_points;
//        y_calc[i] *= 100./total_points;
//        y_adjust[i] *= 100./total_points;
//        y_avg[i] *= 100./total_points;
//    }


//    plt::figure_size(1000, 800);
//    plt::plot(x, y_real);
//    plt::xlim(0, 5);
//    plt::xlabel("Reprojection Error (px)");
//    plt::ylim(0, 15);
//    plt::ylabel("Percentage of Points");
//    plt::title("Real Reprojection");
//    plt::show();
//
//    plt::figure_size(1000, 800);
//    plt::plot(x, y_calc);
//    plt::xlim(0, 5);
//    plt::xlabel("Reprojection Error (px)");
//    plt::ylim(0, 15);
//    plt::ylabel("Percentage of Points");
//    plt::title("Calc Reprojection");
//    plt::show();
//
//    plt::figure_size(1000, 800);
//    plt::plot(x, y_adjust);
//    plt::xlim(0, 5);
//    plt::xlabel("Reprojection Error (px)");
//    plt::ylim(0, 15);
//    plt::ylabel("Percentage of Points");
//    plt::title("Adjust Reprojection");
//    plt::show();
//
//    plt::figure_size(1000, 800);
//    plt::plot(x, y_avg);
//    plt::xlim(0, 5);
//    plt::xlabel("Reprojection Error (px)");
//    plt::ylim(0, 15);
//    plt::ylabel("Percentage of Points");
//    plt::title("Avg Reprojection");
//    plt::show();

    return 0;
}