//
// Created by Cameron Fiore on 12/15/21.
//

#include "../include/sevenScenes.h"
#include "../include/CambridgeLandmarks.h"
#include "../include/functions.h"
#include "../include/Space.h"
#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <algorithm>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include "defines.h"
//#include "fmatrix.h"
//#include "poly1.h"
//#include "poly3.h"
//#include "matrix.h"
//#include "qsort.h"
//#include "svd.h"
//#include "triangulate.h"
//#include "vector.h"

#define PI 3.14159265358979323846
#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"

using namespace std;
using namespace cv;

vector<string> functions::getQueries(const string & queryList, const string & scene) {

    vector<string> queries;
    ifstream file(queryList);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            size_t pos = line.find('/') + 1;
            if (pos > 0 && line.substr(0, pos) == scene) {
                queries.push_back(line);
            }
        }
        file.close();
    } else {
        cout << "Can't open query list file." << endl;
        exit(1);
    }
    return queries;
}

tuple<string, Eigen::Matrix3d,Eigen::Vector3d, vector<double>,
        vector<string>,
        vector<Eigen::Matrix3d>, vector<Eigen::Vector3d>,
        vector<Eigen::Matrix3d>, vector<Eigen::Vector3d>,
        vector<vector<double>>,
        vector<vector<cv::Point2d>>, vector<vector<cv::Point2d>>>
        functions::parseRelposeFile (const string & dir, const string & query, const string & fn) {

    string query_id = query.substr(0, query.find('.'));
    Eigen::Matrix3d R_q;
    Eigen::Vector3d T_q;
    vector<double> K_q;

    vector<string> anchors;
    vector<Eigen::Matrix3d> R_is, R_qis;
    vector<Eigen::Vector3d> T_is, T_qis;
    vector<vector<double>> K_is;
    vector<vector<cv::Point2d>> all_points_q, all_points_i;

    string relpose_fn = dir + query_id + "." + fn + ".txt";
    ifstream file (relpose_fn);
    if (file.is_open()) {
        string first_line;
        if (getline(file, first_line)) {
            stringstream ss (first_line);
            int count = 0;
            string it;
            while (ss >> it) {
                if (count == 0) {
                    assert(it == "Query_Info:");
                } else if (count >= 1 && count <= 9) {
                    int row = (count - 1) / 3;
                    int col = (count - 1) % 3;
                    R_q(row, col) = stod(it);
                } else if (count >= 10 && count <= 12) {
                    int row = count - 10;
                    T_q(row) = stod(it);
                } else {
                    assert(count >= 13 && count <= 16);
                    K_q.push_back(stod(it));
                }
                count++;
            }
        } else {
            cout << "Can't ready query info " + query_id << endl;
            exit(1);
        }

        string line;
        while(getline(file, line)) {
            stringstream ss (line);
            int count = 0;
            int inner = 0;
            double q_x = 0, q_y = 0, i_x = 0, i_y = 0;
            string it;

            string anchor;
            Eigen::Matrix3d R_i, R_qi;
            Eigen::Vector3d T_i, T_qi;
            vector<double> K_i;
            vector<cv::Point2d> points_q, points_i;

            while(ss >> it) {
                if (count == 0) {
                    anchor = it;
                } else if (count >= 1 && count <= 9) {
                    int row = (count - 1) / 3;
                    int col = (count - 1) % 3;
                    R_i(row, col) = stod(it);
                } else if (count >= 10 && count <= 12) {
                    int row = count - 10;
                    T_i(row) = stod(it);
                } else if (count >= 13 && count <= 16) {
                    K_i.push_back(stod(it));
                } else if(count >= 17 && count <= 25){
                    int row = (count - 17) / 3;
                    int col = (count - 17) % 3;
                    R_qi(row, col) = stod(it);
                } else if (count >= 26 && count <= 28) {
                    int row = count - 26;
                    T_qi(row) = stod(it);
                } else {
                    assert(count >= 29);
                    assert(round(stod(it)) - stod(it) == 0);
                    if (inner == 0) {
                        q_x = stod(it);
                    } else if (inner == 1) {
                        q_y = stod(it);
                    } else if (inner == 2) {
                        i_x = stod(it);
                    } else {
                        assert(inner == 3);
                        i_y = stod(it);
                        points_q.emplace_back(q_x, q_y);
                        points_i.emplace_back(i_x, i_y);
                        inner = -1;
                    }
                    inner++;
                }
                count++;
            }
            anchors.push_back(anchor);
            R_is.push_back(R_i);
            T_is.push_back(T_i);
            R_qis.push_back(R_qi);
            T_qis.push_back(T_qi);
            K_is.push_back(K_i);
            all_points_q.push_back(points_q);
            all_points_i.push_back(points_i);
        }
        file.close();
    } else {
        cout << "Can't open relpose file." << endl;
        exit(1);
    }
    auto t = make_tuple(query_id, R_q, T_q, K_q, anchors, R_is, T_is, R_qis, T_qis, K_is, all_points_q, all_points_i);
    return t;
}

vector<int> functions::optimizeSpacingZhou (const vector<Eigen::Vector3d> & old_centers,
                                            double l_thresh, double u_thresh, int N) {
    vector<int> indices {0};
    vector<Eigen::Vector3d> new_centers {old_centers[0]};
    for (int i = 1; i < old_centers.size(); i++) {
        Eigen::Vector3d c = old_centers[i];
        bool too_close = false;
        bool close_enough = false;
        for (auto & center : new_centers) {
            double dist = functions::getDistBetween(c, center);
            if (dist <= u_thresh) close_enough = true;
            if (dist < l_thresh) {
                too_close = true;
                break;
            }
        }
        if(close_enough && !too_close) {
            indices.push_back(i);
            new_centers.push_back(c);
        }
        if (indices.size() == N) break;
    }
    return indices;
}

void functions::get_SG_points(const string & query, const string & db, vector<cv::Point2d> & pts_q, vector<cv::Point2d> & pts_i) {
    string path = query.substr(query.find("data/") + 5, query.find(".color.png"));
    path = "/Users/cameronfiore/Python/Localization/SuperGluePretrainedNetwork/7-Scenes/" + path + "/cpp_text_files/";
    string seq = functions::getSequence(db);
    string im = db.substr(db.find("frame"), db.find(".color"));
    string mf = path + "seq-" + seq + "_" + im + "_matches.txt";

    ifstream file (mf);
    if (file.is_open()) {
        string line;
        while (getline(file, line))
        {
            stringstream ss(line);
            double q_x, q_y, db_x, db_y;
            for (int i = 0; i < 4; i++) {
                if( i == 0) ss >> q_x;
                if( i == 1) ss >> q_y;
                if( i == 2) ss >> db_x;
                if( i == 3) ss >> db_y;
            }
            pts_q.emplace_back(q_x, q_y);
            pts_i.emplace_back(db_x, db_y);
        }
        file.close();
    }
}

void functions::record_spaced(const string & query, const vector<string> & spaced, const string & folder) {

    ofstream file (folder + "/pairs.txt");
    if (file.is_open()) {
        for (const auto & im : spaced) {
            file << query << ".color.png    " << im << ".color.png" << endl;
        }
        file.close();
    }
}


Eigen::Matrix3d functions::smallRandomRotationMatrix(double s) {

        random_device generator;
//        default_random_engine engine;
//        engine.seed(1);
        uniform_real_distribution<double> uniform (0., 2*PI);
        normal_distribution<double> gaussian (0., s);

        double alpha = uniform(generator);
        double beta = uniform(generator);
        double gamma = gaussian(generator);
//
//        double alpha = uniform(engine);
//        double beta = uniform(engine);
//        double gamma = gaussian(engine);

        double z = sin(alpha);
        double h = abs(cos(alpha));

        double x = h * cos(beta);
        double y = h * sin(beta);

        Eigen::Vector3d vec {x, y, z};
        vec.normalize();

        double n = vec.norm();

        Eigen::AngleAxis<double> aa (gamma, vec);

        Eigen::Matrix3d r = aa.toRotationMatrix();

        double d = r.determinant();

        return r;
}

pair<vector<cv::Point2d>, vector<cv::Point2d>> functions::same_inliers(
        const vector<cv::Point2d> & pts_q, const vector<cv::Point2d> & pts_db,
        const Eigen::Matrix3d & R_qk, const Eigen::Vector3d & T_qk,
        const Eigen::Matrix3d & R_kq, const Eigen::Vector3d & T_kq,
        const Eigen::Matrix3d & K) {

    vector<cv::Point2d> inliers_q, inliers_db;

    vector<cv::Point2d> pts_q_inliers_qk;
    vector<cv::Point2d> pts_db_inliers_qk;

    Eigen::Matrix3d t_qk_cross{
            {0,        -T_qk(2), T_qk(1)},
            {T_qk(2),  0,        -T_qk(0)},
            {-T_qk(1), T_qk(0),  0}};

    Eigen::Matrix3d E_qk = t_qk_cross * R_qk;

    for (int i = 0; i < pts_q.size(); i++) {

        Eigen::Vector3d pt_q{pts_q[i].x, pts_q[i].y, 1.};
        Eigen::Vector3d pt_db{pts_db[i].x, pts_db[i].y, 1.};

        Eigen::Vector3d epiline = K.inverse().transpose() * E_qk * K.inverse() * pt_db;

        double dist = abs(epiline(0) * pt_q(0) + epiline(1) * pt_q(1) + epiline(2)) /
                      sqrt(epiline(0) * epiline(0) + epiline(1) * epiline(1));

        if (dist <= .1) {
            pts_q_inliers_qk.push_back(pts_q[i]);
            pts_db_inliers_qk.push_back(pts_db[i]);
        }
    }

    vector<cv::Point2d> pts_q_inliers_kq;
    vector<cv::Point2d> pts_db_inliers_kq;

    Eigen::Matrix3d t_kq_cross{
            {0,        -T_kq(2), T_kq(1)},
            {T_kq(2),  0,        -T_kq(0)},
            {-T_kq(1), T_kq(0),  0}};

    Eigen::Matrix3d E_kq = t_kq_cross * R_kq;

    for (int i = 0; i < pts_q.size(); i++) {

        Eigen::Vector3d pt_q{pts_q[i].x, pts_q[i].y, 1.};
        Eigen::Vector3d pt_db{pts_db[i].x, pts_db[i].y, 1.};

        Eigen::Vector3d epiline = K.inverse().transpose() * E_kq * K.inverse() * pt_q;

        double dist = abs(epiline(0) * pt_db(0) + epiline(1) * pt_db(1) + epiline(2)) /
                      sqrt(epiline(0) * epiline(0) + epiline(1) * epiline(1));

        if (dist <= .1) {
            pts_q_inliers_kq.push_back(pts_q[i]);
            pts_db_inliers_kq.push_back(pts_db[i]);
        }
    }

    vector<cv::Point2d> pts_q_inliers_kq_diff = pts_q_inliers_kq;
    vector<cv::Point2d> pts_q_inliers_qk_diff;

    for (int i = 0; i < pts_q_inliers_qk.size(); i++) {
        bool matched = false;
        for (int j = 0; j < pts_q_inliers_kq_diff.size(); j++) {
            double dist = sqrt(pow((pts_q_inliers_qk[i].x - pts_q_inliers_kq_diff[j].x), 2.)
                               + pow((pts_q_inliers_qk[i].y - pts_q_inliers_kq_diff[j].y), 2.));
            if (dist <= 0.000001) {
                pts_q_inliers_kq_diff.erase(pts_q_inliers_kq_diff.begin() + j);
                inliers_q.push_back(pts_q_inliers_qk[i]);
                matched = true;
                break;
            }
        }
        if (!matched) {
            pts_q_inliers_qk_diff.push_back(pts_q_inliers_qk[i]);
        }
    }

    vector<cv::Point2d> pts_db_inliers_kq_diff = pts_db_inliers_kq;
    vector<cv::Point2d> pts_db_inliers_qk_diff;

    for (int i = 0; i < pts_db_inliers_qk.size(); i++) {
        bool matched = false;
        for (int j = 0; j < pts_db_inliers_kq_diff.size(); j++) {
            double dist = sqrt(pow((pts_db_inliers_qk[i].x - pts_db_inliers_kq_diff[j].x), 2.)
                               + pow((pts_db_inliers_qk[i].y - pts_db_inliers_kq_diff[j].y), 2.));
            if (dist <= 0.000001) {
                pts_db_inliers_kq_diff.erase(pts_db_inliers_kq_diff.begin() + j);
                inliers_db.push_back(pts_db_inliers_qk[i]);
                matched = true;
                break;
            }
        }
        if (!matched) {
            pts_db_inliers_qk_diff.push_back(pts_db_inliers_qk[i]);
        }
    }
    int x = 0;
    return {inliers_q, inliers_db};
}

double functions::triangulateRays(const Eigen::Vector3d &ci, const Eigen::Vector3d &di, const Eigen::Vector3d &cj,
                                   const Eigen::Vector3d &dj, Eigen::Vector3d &intersect) {
    Eigen::Vector3d dq = di.cross(dj);
    Eigen::Matrix3d D;
    D.col(0) = di;
    D.col(1) = -dj;
    D.col(2) = dq;
    Eigen::Vector3d b = cj - ci;
    Eigen::Vector3d sol = D.colPivHouseholderQr().solve(b);
    intersect = ci + sol(0) * di + (sol(2) / 2) * dq;
    Eigen::Vector3d toReturn = sol(2) * dq;
    return toReturn.norm();
}

double functions::getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
    double x = p1(0) - p2(0);
    double y = p1(1) - p2(1);
    double z = p1(2) - p2(2);
    return sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
}

double functions::getPoint3DLineDist(const Eigen::Vector3d & h, const Eigen::Vector3d & c_k, const Eigen::Vector3d & v_k) {
    double numerator = (h - c_k).cross(h - (c_k + v_k)).norm();
    double denominator = v_k.norm();
    return numerator/denominator;
}

double functions::getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2) {
    double dot = d1.dot(d2);
    double d1_norm = d1.norm();
    double d2_norm = d2.norm();
    double frac = dot / (d1_norm * d2_norm);
    if (abs(frac) > 1. && abs(frac) < 1.0000000001) frac = round(frac);
    double rad = acos(frac);
    return rad * 180.0 / PI;
}

double functions::rotationDifference(const Eigen::Matrix3d & r1, const Eigen::Matrix3d & r2) {

    Eigen::Matrix3d R12 = r1 * r2.transpose();
    double trace = R12.trace();
    if (trace > 3. && trace < 3.0000000001) trace = 3.;
    double theta = acos((trace-1.)/2.);
    theta = theta * 180. / PI;
    return theta;
}

vector<double> functions::getReprojectionErrors(const vector<cv::Point2d> & pts_to, const vector<cv::Point2d> & pts_from,
                                     const Eigen::Matrix3d & K, const Eigen::Matrix3d & R, const Eigen::Vector3d & T) {
    vector<double> to_return(301);

    Eigen::Vector3d T_normalized = T.normalized();
    for (int i = 0; i < pts_from.size(); i++) {

        Eigen::Vector3d pt_from {pts_from[i].x, pts_from[i].y, 1.};
        Eigen::Vector3d pt_to {pts_to[i].x, pts_to[i].y, 1.};

        Eigen::Matrix3d T_cross {
                {0,               -T_normalized(2), T_normalized(1)},
                {T_normalized(2),  0,              -T_normalized(0)},
                {-T_normalized(1), T_normalized(0),               0}};
        Eigen::Matrix3d E = T_cross * R;

        Eigen::Vector3d ep_line = K.inverse().transpose() * E * K.inverse() * pt_from;

        double error = abs(ep_line(0) * pt_to(0) + ep_line(1) * pt_to(1) + ep_line(2)) /
                sqrt(ep_line(0) * ep_line(0) + ep_line(1) * ep_line(1));

        int idx = int (round(error*100));

        if (idx < 301) to_return[idx]++;
    }
    return to_return;
}

pair<cv::Mat, vector<cv::KeyPoint>> functions::getDescriptors(const string & image, const string & ext, const string & method) {

    vector<KeyPoint> kp_vec;
    Mat desc;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image_U, gray;
        imread(image + ext, IMREAD_COLOR).copyTo(image_U);
        cvtColor(image_U, gray, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray, Mat(), kp_vec, desc);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image_U, gray;
        imread(image + ext, IMREAD_COLOR).copyTo(image_U);
        cvtColor(image_U, gray, COLOR_RGB2GRAY);
        surf->detectAndCompute(gray, Mat(), kp_vec, desc);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();
        Mat im = imread(image + ext, IMREAD_GRAYSCALE);
        sift->detectAndCompute(im, Mat(), kp_vec, desc);
    } else {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    return {desc, kp_vec};
}

bool functions::findMatches(double ratio,
                 const cv::Mat & desc_i, const vector<cv::KeyPoint> & kp_i,
                 const cv::Mat & desc_q, const vector<cv::KeyPoint> & kp_q,
                 vector<cv::Point2d> & pts_i, vector<cv::Point2d> & pts_q) {

    auto matcher = BFMatcher::create();
    vector<vector<DMatch>> matches_2nn_qi;
    matcher->knnMatch(desc_q, desc_i, matches_2nn_qi, 2);

    for (const auto & match_qi: matches_2nn_qi) {
        double r = match_qi[0].distance / match_qi[1].distance;
        if (r <= ratio) {
            pts_q.push_back(kp_q[match_qi[0].queryIdx].pt);
            pts_i.push_back(kp_i[match_qi[0].trainIdx].pt);
        }
    }

    if (pts_q.empty()) return false;

    return true;
}

bool functions::findMatches(const string &db_image, const string &query_image, const string & ext, const string &method, double ratio,
                             vector<Point2d> &pts_db, vector<Point2d> &pts_query) {


    vector<KeyPoint> kp_vec1, kp_vec2;
    Mat desc1, desc2;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray1, Mat(), kp_vec1, desc1);
        orb->detectAndCompute(gray2, Mat(), kp_vec2, desc2);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        surf->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        surf->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();

        Mat image1 = imread(query_image + ext);
        Mat image2 = imread(db_image + ext);
        sift->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        sift->detectAndCompute(image2, Mat(), kp_vec2, desc2);

//        cv::Mat kp_im1;
//        cv::drawKeypoints(image1, kp_vec1, kp_im1);
//        cv::imshow("Query Keypoints", kp_im1);
//        cv::waitKey(0);
//
//        cv::Mat kp_im2;
//        cv::drawKeypoints(image2, kp_vec2, kp_im2);
//        cv::imshow("Database Keypoints", kp_im2);
//        cv::waitKey(0);
    } else {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    auto matcher = BFMatcher::create(NORM_L2, false);
    vector< vector<DMatch> > matches_2nn_12, matches_2nn_21;
    matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
    vector<Point2d> selected_points1, selected_points2;

    for (const auto & m : matches_2nn_12) { // i is queryIdx
        if( m[0].distance/m[1].distance < ratio
            and
            matches_2nn_21[m[0].trainIdx][0].distance
            / matches_2nn_21[m[0].trainIdx][1].distance < ratio )
        {
            if(matches_2nn_21[m[0].trainIdx][0].trainIdx
               == m[0].queryIdx)
            {
                selected_points1.push_back(kp_vec1[m[0].queryIdx].pt);
                selected_points2.push_back(kp_vec2[matches_2nn_21[m[0].trainIdx][0].queryIdx].pt);
            }
        }
    }

    pts_query = selected_points1;
    pts_db = selected_points2;

    if (pts_db.empty()) {
        return false;
    }

    return true;
}

bool functions::findMatchesSorted(const string &db_image, const string & ext, const string &query_image, const string &method, double ratio,
                                  vector<tuple<Point2d, Point2d, double>> & points) {


    vector<KeyPoint> kp_vec1, kp_vec2;
    Mat desc1, desc2;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray1, Mat(), kp_vec1, desc1);
        orb->detectAndCompute(gray2, Mat(), kp_vec2, desc2);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + ext, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + ext, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        surf->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        surf->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();
        Mat image1 = imread(query_image + ext);
        Mat image2 = imread(db_image + ext);
        sift->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        sift->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    auto matcher = BFMatcher::create(NORM_L2, false);
    vector< vector<DMatch> > matches_2nn_12, matches_2nn_21;
    matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
    vector<tuple<Point2d, Point2d, double>> selected_points; // query, db, match_ratio

    for (const auto & m : matches_2nn_12) {
        double r = m[0].distance/m[1].distance;
        if (r < ratio and matches_2nn_21[m[0].trainIdx][0].distance/matches_2nn_21[m[0].trainIdx][1].distance < ratio )
        {
            if(matches_2nn_21[m[0].trainIdx][0].trainIdx == m[0].queryIdx)
            {
                selected_points.emplace_back(kp_vec1[m[0].queryIdx].pt,
                                             kp_vec2[matches_2nn_21[m[0].trainIdx][0].queryIdx].pt,
                                             r);
            }
        }
    }

    if (selected_points.empty()) return false;

    sort(selected_points.begin(), selected_points.end(), [](const auto & lhs, const auto & rhs){
        return get<2>(lhs) < get<2>(rhs);
    });
    points = selected_points;
    return true;
}

void functions::findInliers(const Eigen::Matrix3d & R_q,
                            const Eigen::Vector3d & T_q,
                            const Eigen::Matrix3d & R_k,
                            const Eigen::Vector3d & T_k,
                            const Eigen::Matrix3d & K,
                            double threshold,
                            vector<cv::Point2d> & pts_q,
                            vector<cv::Point2d> & pts_db) {

    Eigen::Matrix3d R_qk = R_q * R_k.transpose();
    Eigen::Vector3d T_qk = T_q - R_qk * T_k;
    T_qk.normalize();
    Eigen::Matrix3d T_qk_cross{{0, -T_qk(2), T_qk(1)},
                               {T_qk(2), 0, -T_qk(0)},
                               {-T_qk(1), T_qk(0), 0}};
    Eigen::Matrix3d E_qk = T_qk_cross * R_qk;
    Eigen::Matrix3d F_qk = K.inverse().transpose() * E_qk * K.inverse();

    vector<cv::Point2d> inliers_q, inliers_db;
    for (int i = 0; i < pts_q.size(); i++) {
        Eigen::Vector3d pt_q {pts_q[i].x, pts_q[i].y, 1.};
        Eigen::Vector3d pt_db {pts_db[i].x, pts_db[i].y, 1.};

        Eigen::Vector3d epiline = F_qk * pt_db;

        double error = abs(epiline[0] * pt_q[0] + epiline[1] * pt_q[1] + epiline[2]) /
                       sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

        if (error <= threshold) {
            inliers_q.push_back(pts_q[i]);
            inliers_db.push_back(pts_db[i]);
        }
    }

    pts_q = inliers_q;
    pts_db = inliers_db;
}

int functions::getRelativePose(const string & db_image, const string & ext, const string & query_image, const double * K, const string & method,
                               Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
    try {
        vector<Point2d> pts_db, pts_q;
        if (!findMatches(db_image, ext, query_image, method, 0.8, pts_db, pts_q)) return 0;

        Mat im = imread(db_image + ext);
        Mat mask;
        Mat K_mat = (Mat_<double>(3, 3) <<
                K[0], 0., K[2],
                0., K[1], K[3],
                0., 0.,   1.);
        Mat E_kq = findEssentialMat(pts_db,pts_q,K_mat, RANSAC, 0.999999, 3.0, mask);

        vector<Point2d> inlier_db_points, inlier_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                inlier_db_points.push_back(pts_db[i]);
                inlier_q_points.push_back(pts_q[i]);
            }
        }

        mask.release();
        Mat R, t;
        recoverPose(E_kq,inlier_db_points,inlier_q_points,K_mat, R, t, mask);

        vector<Point2d> recover_db_points, recover_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                recover_db_points.push_back(inlier_db_points[i]);
                recover_q_points.push_back(inlier_q_points[i]);
            }
        }

        cv2eigen(R, R_kq);
        cv2eigen(t, t_kq);

        return int (recover_db_points.size());
    } catch (...) {
        return 0;
    }
}

bool functions::getRelativePose(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q,
                                const double * K,
                                double match_thresh,
                                Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
    try {
        Mat mask;
        Mat K_mat = (Mat_<double>(3, 3) << K[0], 0., K[2], 0., K[1], K[3], 0., 0., 1.);

        Mat E_qk_sols = findEssentialMat(pts_db, pts_q, K_mat, cv::RANSAC, 0.9999999999, match_thresh, mask);
        Mat E_qk = E_qk_sols(cv::Range(0, 3), cv::Range(0, 3));

        vector<Point2d> inlier_db_points, inlier_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                inlier_db_points.push_back(pts_db[i]);
                inlier_q_points.push_back(pts_q[i]);
            }
        }

        mask.release();
        Mat R, t;
        recoverPose(E_qk, inlier_db_points, inlier_q_points, K_mat, R, t, mask);

        vector<Point2d> recover_db_points, recover_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                recover_db_points.push_back(inlier_db_points[i]);
                recover_q_points.push_back(inlier_q_points[i]);
            }
        }

        cv2eigen(R, R_kq);
        cv2eigen(t, t_kq);

        pts_db = recover_db_points;
        pts_q = recover_q_points;
        return true;
    } catch (...) {
        return false;
    }
}

int functions::getRelativePose3D(const string &db_image, const string & ext, const string &query_image, const string &method,
                                Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
//    try {
    vector<Point2d> pts_db, pts_q;
    if (!findMatches(db_image, ext, query_image, method, 0.7, pts_db, pts_q)) { return 0; }

    vector<Point3d> pts_db_3d;
    vector<Point2d> pts_q_2d;
    Mat depth = imread(db_image + ".depth.png");
    for (int i = 0; i < pts_db.size(); i++) {
        Point3d pt_3d;
        if (sevenScenes::get3dfrom2d(pts_db[i], depth, pt_3d)) {
            pts_db_3d.push_back(pt_3d);
            pts_q_2d.push_back(pts_q[i]);
        }
    }

    Mat im = imread(db_image + ext);
    Mat mask;
    Mat K = (Mat_<double>(3, 3) <<
                                        585.6, 0., 316.,
                                        0., 585.6, 247.6,
                                        0., 0., 1.);
    Mat R, Rvec, t;

    Mat distCoeffs = Mat();
    solvePnPRansac(pts_db_3d, pts_q_2d, K,
                       distCoeffs, Rvec, t, true,
                       1000, 8.0, 0.999, mask);
    Rodrigues(Rvec, R);


    int inlier_points = 0;
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<unsigned char>(i)) {
            inlier_points++;
        }
    }

    cv2eigen(R, R_kq);
    cv2eigen(t, t_kq);

    return inlier_points;
//    } catch (...) {
//        return 0;
//    }
}

double functions::drawLines(Mat & im1, Mat & im2,
               const vector<Point3d> & lines_draw_on_1,
               const vector<Point3d> & lines_draw_on_2,
               const vector<Point2d> & pts1,
               const vector<Point2d> & pts2,
               const vector<Scalar> & colors) {
    int c = im1.cols;
    double total_rep_error = 0.;
    for (int i = 0; i < pts1.size(); i++) {
        Point3d l1 = lines_draw_on_1[i];
        Point3d l2 = lines_draw_on_2[i];

        Point2d pt1 = pts1[i];
        Point2d pt2 = pts2[i];
        Scalar color = colors[i];

        Point2d i1 (0., -l1.z/l1.y);
        Point2d j1 (c, -(l1.z+l1.x*c)/l1.y);
        double error = abs(l1.x*pt1.x + l1.y*pt1.y + l1.z) / sqrt(l1.x*l1.x + l1.y*l1.y);
        total_rep_error += error;

        Point2d i2 (0., -l2.z/l2.y);
        Point2d j2 (c, -(l2.z+l2.x*c)/l2.y);

        line(im1, i1, j1, color);
        line(im2, i2, j2, color);

        circle(im1, pt1, 4, color, -1);
        circle(im2, pt2, 4, color, -1);
    }
    return total_rep_error / double (pts1.size());
}

vector<int> functions::optimizeSpacing(const Eigen::Vector3d & query,
                                          const vector<Eigen::Vector3d> & centers,
                                          int N, bool show_process) {
    Space space (query, centers);
    space.getOptimalSpacing(N, show_process);
    vector<int> ids = space.getIDs();
    return ids;
}

vector<string> functions::randomSelection(const vector<string> & images, int N) {

    if (images.size() <= N) return images;

    vector<string> take = images, give;
    random_device gen;
    while(give.size() < N) {
        uniform_int_distribution<int> d (0, int(take.size()) - 1);
        int idx = d(gen);
        give.push_back(take[idx]);
        take.erase(take.begin() + idx);
    }
    return give;
}


void functions::showKmeans(const vector<pair<Eigen::Vector3d, vector<pair<string, Eigen::Vector3d>>>> & clusters,
                           const vector<cv::Scalar> & colors) {

    double height = 2000.;
    double width = 3000.;
    double max_radius = 50.;

    Eigen::Vector3d total {0, 0, 0};
    int count = 0;
    for (const auto & c : clusters) {
        for (const auto & im : c.second) {
            total += im.second;
            count++;
        }
    }

    Eigen::Vector3d average = total / count;
    double avg_width = average[0];
    double avg_length = average[2];

    double farthest_x = 0.;
    double farthest_y = 0.;
    double max_z = -1000;
    double min_z = 1000;
    for (const auto & c : clusters) {
        for (const auto & im: c.second) {
            double x_dist = abs(avg_width - im.second[0]);
            double y_dist = abs(avg_length - im.second[2]);
            double h = im.second[1];
            if (x_dist > farthest_x) farthest_x = x_dist;
            if (y_dist > farthest_y) farthest_y = y_dist;
            if (h > max_z) {
                max_z = h;
            } else if (h < min_z) {
                min_z = h;
            }
        }
    }

    double m_over_px_x = (farthest_x / (width / 2. - max_radius));
    double m_over_px_y = (farthest_y / (height / 2. - max_radius));
    double m_over_px_z = (max_z - min_z) * 1.2 / max_radius;
    double sep = max_z - min_z;

    cv::Mat canvasImage(int(height), int(width), CV_8UC3, cv::Scalar(255., 255., 255.));

    cv::Point2d im_center {width / 2, height / 2};
    for (int i = 0; i < clusters.size(); i++) {
        cv::Scalar color = colors[i];
        for (const auto & im: clusters[i].second) {
            cv::Point2d c_pt {
                im_center.x + ((im.second[0] - avg_width) / m_over_px_x),
                im_center.y + ((im.second[2] - avg_length) / m_over_px_y)
            };
            double radius = (im.second[1] - min_z + sep*.2) / m_over_px_z;


            cv::circle(canvasImage, c_pt, int(radius), color, -1);
        }
    }
    imshow("KMEANS", canvasImage);
    cv::waitKey(0);
}


//// THIS METHOD IS TAILORED FOR 7 SCENES
vector<string> functions::kMeans(const vector<string> & images, int N, bool show_process) {

    if (images.size() <= N) return images;

    // Get all image centers
    vector<pair<string, Eigen::Vector3d>> im2c;
    im2c.reserve(images.size());
    for (const auto &im: images) {
        im2c.emplace_back(im, sevenScenes::getT(im));
    }

    // Initialize clusters
    vector<pair<string, Eigen::Vector3d>> take = im2c;
    vector<pair<Eigen::Vector3d, vector<pair<string, Eigen::Vector3d>>>> clusters;
    vector<cv::Scalar> colors;
    uniform_int_distribution<int> color_dist(0, 255);
    random_device gen;
    while (clusters.size() < N) {
        uniform_int_distribution<int> d(0, int(take.size()) - 1);
        int idx = d(gen);
        clusters.emplace_back(take[idx].second, vector<pair<string, Eigen::Vector3d>> ());
        take.erase(take.begin() + idx);
        cv::Scalar color(color_dist(gen), color_dist(gen), color_dist(gen));
        colors.push_back(color);
    }

    bool first = true;
    double total_shift = 1;
    while (total_shift > 0.001) {

        for (auto &c: clusters) {
            c.second.clear();
        }

        // Assign to centers
        for (int i = 0; i < im2c.size(); i++) {
            double smallest_dist = 10000;
            int closest = 0;
            vector<int> ties;
            for (int c = 0; c < clusters.size(); c++) {
                double dist = functions::getDistBetween(im2c[i].second, clusters[c].first);

                if (dist == smallest_dist) {
                    ties.push_back(c);
                } else if (dist < smallest_dist) {
                    smallest_dist = dist;
                    closest = i;
                    ties.clear();
                    ties.push_back(c);
                }
            }

            int smallest = 0;
            int smallest_size = 10000;
            for (const auto & c : ties) {
                int s = int(clusters[c].second.size());
                if (s < smallest_size) {
                    smallest_size = s;
                    smallest = c;
                }
            }

            clusters[smallest].second.push_back(im2c[i]);
        }

        // Recalc centers
        total_shift = 0;
        for (auto &c: clusters) {
            Eigen::Vector3d new_center{0, 0, 0};
            for (const auto &im: c.second) {
                new_center += im.second;
            }
            new_center /= double(c.second.size());
            total_shift += functions::getDistBetween(new_center, c.first);
            c.first = new_center;
        }

        // Display Center
        if (show_process) {
            showKmeans(clusters, colors);
        }
    }

    // Find closest image to cluster center
    vector<string> to_return;
    to_return.reserve(clusters.size());
    for (const auto &c: clusters) {
        double smallest_dist = 10000;
        string closest;
        for (const auto &im: c.second) {
            double dist = functions::getDistBetween(c.first, im.second);
            if (dist < smallest_dist) {
                smallest_dist = dist;
                closest = im.first;
            }
        }
        to_return.push_back(closest);
    }

    return to_return;
}



////SURF functions
//void functions::loadFeaturesSURF(const vector<string> &listImage, vector<vector<vector<float>>> &features)
//{
//    features.clear();
//    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
//
//    for (const string& im : listImage)
//    {
//        Mat image = imread(im + ext, IMREAD_GRAYSCALE);
//        if (image.empty())
//        {
//            cout << "The image " << im << "is empty" << endl;
//            exit(1);
//        }
//        Mat mask;
//        vector<KeyPoint> keyPoints;
//        vector<float> descriptors;
//
//        surf->detectAndCompute(image, mask, keyPoints, descriptors);
//        features.push_back(vector<vector<float>>());
//        changeStructureSURF(descriptors, features.back(), surf->descriptorSize());
//
//        cout << im << endl;
//    }
//    cout << "Done!" << endl;
//}
//
//void functions::changeStructureSURF(const vector<float> &plain, vector<vector<float>> &out, int L)
//{
//    out.resize(plain.size() / L);
//    unsigned int j = 0;
//
//    for(unsigned int i = 0; i < plain.size(); i += L, ++j)
//    {
//        out[j].resize(L);
//        std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
//    }
//}
//
//void functions::DatabaseSaveSURF(const vector<vector<vector<float>>> &features)
//{
//    Surf64Vocabulary voc;
//
//    cout << "Creating vocabulary..." << endl;
//    voc.create(features);
//    cout << "... done!" << endl;
//
//    cout << "Vocabulary information: " << endl << voc << endl;
//
//    cout << "Saving vocabulary..." << endl;
//    string vocFile = "voc.yml.gz";
//    voc.save(FOLDER + vocFile);
//    cout << "Done" << endl;
//    cout << "Creating database..." << endl;
//
//    Surf64Database db(voc, false, 0);
//
//    for(const auto & feature : features)
//    {
//        db.add(feature);
//    }
//
//    cout << "... done!" << endl;
//
//    cout << "Database information: " << endl << db << endl;
//
//    cout << "Saving database..." << endl;
//    string dbFile = "db.yml.gz";
//    db.save(FOLDER + dbFile);
//    cout << "... done!" << endl;
//
//}


//// ORB functions
//void functions::loadFeaturesORB(const vector<string> &listImage, vector<vector<Mat>> &features)
//{
//    features.clear();
//    Ptr<ORB> orb = ORB::create();
//
//    for (const string& im : listImage)
//    {
//        Mat image = imread(im + ext, IMREAD_GRAYSCALE);
//        if (image.empty())
//        {
//            cout << "The image " << im << "is empty" << endl;
//            exit(1);
//        }
//        Mat mask;
//        vector<KeyPoint> keyPoints;
//        Mat descriptors;
//
//        orb->detectAndCompute(image, mask, keyPoints, descriptors);
//        features.emplace_back();
//        changeStructureORB(descriptors, features.back());
//
//        cout << im << endl;
//    }
//    cout << "Done!" << endl;
//}
//
//void functions::changeStructureORB(const Mat &plain, vector<Mat> &out)
//{
//    out.resize(plain.rows);
//
//    for(int i = 0; i < plain.rows; ++i)
//    {
//        out[i] = plain.row(i);
//    }
//}
//
//void functions::DatabaseSaveORB(const vector<vector<Mat>> &features)
//{
//    OrbVocabulary voc;
//
//    cout << "Creating vocabulary..." << endl;
//    voc.create(features);
//    cout << "... done!" << endl;
//
//    cout << "Vocabulary information: " << endl << voc << endl;
//
//    cout << "Saving vocabulary..." << endl;
//    string vocFile = "voc_orb.yml.gz";
//    voc.save(FOLDER + vocFile);
//    cout << "Done" << endl;
//    cout << "Creating database..." << endl;
//
//    OrbDatabase db(voc, false, 0);
//
//    for(const auto & feature : features)
//    {
//        db.add(feature);
//    }
//
//    cout << "... done!" << endl;
//
//    cout << "Database information: " << endl << db << endl;
//
//    cout << "Saving database..." << endl;
//    string dbFile = "db_orb.yml.gz";
//    db.save(FOLDER + dbFile);
//    cout << "... done!" << endl;
//
//}

// surf
// orb
// optional _jts meaning "just this scene"
//void functions::saveResultVector(const vector<pair<int, float>>& result, const string& query_image, const string& method, const string & num)
//{
//    ofstream file (query_image + "." + method + ".top" + num + ".txt");
//    if(file.is_open())
//    {
//        for (const pair<int, float>& r : result)
//        {
//            file << to_string(r.first) + "  " + to_string(r.second) + "\n";
//        }
//        file.close();
//    }
//}
//
//vector<pair<int, float>> functions::getResultVector(const string& query_image, const string& method, const string & num)
//{
//    vector<pair<int, float>> result;
//    ifstream file (query_image + "." + method + ".top" + num + ".txt");
//    if(file.is_open())
//    {
//        string line;
//        while (getline(file, line))
//        {
//            stringstream ss (line);
//            string n;
//            vector<string> temp;
//            while(ss >> n)
//            {
//                temp.push_back(n);
//            }
//            result.emplace_back(stoi(temp[0]), stof(temp[1]));
//        }
//        file.close();
//    }
//    return result;
//}

string functions::getScene(const string & image, const string & mod) {

    if(mod == "Street") {
        string scene = image;
        scene.erase(0, scene.find("data/") + 5);
        scene = scene.substr(0, scene.find("/img"));
        return scene;
    }

    string scene = image;
    scene.erase(0, scene.find("data/") + 5);
    scene = scene.substr(0, scene.find("/seq"));
    return scene;
}

string functions::getSequence(const string & image) {
    string seq = image;
    string scene = functions::getScene(image, "");
    seq.erase(0, seq.find(scene) + scene.size() + 5);
    seq = seq.substr(0, seq.find("/frame"));
    return seq;
}

vector<string> functions::getTopN(const string& query_image, const string & ext, int N) {
    vector<string> result;
    ifstream file (query_image + ext + ".1000nn.txt");
    if(file.is_open())
    {
        string line;
        int n = 0;
        while (n < N)
        {
            getline(file, line);
            stringstream ss (line);
            string im;
            string fn;
            if (ss >> im)
            {
                fn = im.substr(0, im.find(ext));
            }
            result.push_back(fn);
            n++;
        }
    }
    return result;
}

void functions::retrieveSimilar(const string & query_image,
                                const string & replace,
                                const string & ext,
                                int max_num,
                                double max_dist,
                                vector<string> & similar,
                                vector<double> & distances) {

    ifstream file (query_image + ext + ".1000nn.txt");

    if (file.is_open()) {
        string line;
        while (similar.size() < max_num && getline(file, line)) {
            stringstream ss(line);
            string buf, fn;
            double dist;
            int count = 0;
            while (ss >> buf) {
                if (count == 0) {
                    if (replace.empty()) {
                        fn = buf.substr(0, buf.find(ext));
                    } else {
                        size_t start = buf.find("data")+4;
                        size_t stop = buf.length() - start - ext.length();
                        fn = replace + buf.substr(start, stop);
                    }
                } else {
                    dist = stod(buf);
                }
                count++;
            }
            if (dist <= max_dist) {
                similar.push_back(fn);
                distances.push_back(dist);
            }
        }
        file.close();
    }
}

map<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<double>, vector<double>, vector<pair<cv::Point2d, cv::Point2d>>>>
        functions::getRelativePoses(const string & query, const string & data_folder) {

    map<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<double>, vector<double>, vector<pair<cv::Point2d, cv::Point2d>>>> map;

    string seq = query.substr(query.find("seq"),6);
    string image_base = query.substr(query.find("frame"),12);
    string fn = data_folder + "relposes/" + seq + "/" + image_base + "_rel_poses.txt";
    ifstream poses (fn);
    if (poses.is_open()) {
        string line;
        while(getline(poses, line)) {
            stringstream ss(line);

            string q;
            ss >> q;
            string cur_query = data_folder + q;
            assert(cur_query == query);


            string im;
            ss >> im;
            string image = data_folder + im;

            string R00, R01, R02, R10, R11, R12, R20, R21, R22;
            ss >> R00;
            ss >> R01;
            ss >> R02;
            ss >> R10;
            ss >> R11;
            ss >> R12;
            ss >> R20;
            ss >> R21;
            ss >> R22;
            Eigen::Matrix3d R_qk{{stod(R00), stod(R01), stod(R02)},
                                 {stod(R10), stod(R11), stod(R12)},
                                 {stod(R20), stod(R21), stod(R22)}};
            string T0, T1, T2;
            ss >> T0;
            ss >> T1;
            ss >> T2;
            Eigen::Vector3d T_qk{stod(T0), stod(T1), stod(T2)};

            string cx, cy, f;
            ss >> cx;
            ss >> cy;
            ss >> f;
            ss >> f;
            vector<double> K1{stod(cx), stod(cy), stod(f), stod(f)};

            ss >> cx;
            ss >> cy;
            ss >> f;
            ss >> f;
            vector<double> K2{stod(cx), stod(cy), stod(f), stod(f)};

            int counter = 1;
            string val;
            vector<pair<cv::Point2d, cv::Point2d>> points;
            double x_q, y_q, x_i, y_i;
            while (ss >> val) {
                if (counter == 1) {
                    x_q = stod(val);
                } else if (counter == 2) {
                    y_q = stod(val);
                } else if (counter == 3) {
                    x_i = stod(val);
                } else {
                    y_i = stod(val);
                    cv::Point2d pq(x_q, y_q), pi(x_i, y_i);
                    points.emplace_back(pq, pi);
                    counter = 0;
                }
                counter++;
            }

            tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<double>, vector<double>, vector<pair<cv::Point2d, cv::Point2d>>> pp = make_tuple(
                    R_qk, T_qk, K1, K2, points);
            pair<string, tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<double>, vector<double>, vector<pair<cv::Point2d, cv::Point2d>>>> item = make_pair(
                    image, pp);
            map.insert(item);
        }
        poses.close();
    }
    return map;
}





// Visualize

void functions::showTop(int rows, int cols,
                        const string & query_image, const vector<string> & returned,
                        const string & ext, const string & title) {\

    int num = int(returned.size());

//    Mat q = imread(query_image + ext);
//    imshow(query_image, q);
//    waitKey();

    string scene = getScene(query_image, "");

    vector<pair<Mat, string>> vecMat;
    vecMat.reserve(num);
    for(const auto & p : returned) {
        vecMat.emplace_back(imread(p + ext), p);
    }


    // Get window parameters
    int nRows = rows;
    int windowHeight = 6000;
    int edgeThickness = 20;
    int imagesPerRow = cols;
    int resizeHeight = int (floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0))) - edgeThickness;
    int maxRowLength = 0;
    std::vector<int> resizeWidth;
    for (int k = 0, i = 0; i < nRows; i++) {
        int thisRowLen = 0;
        for (int j = 0; j < imagesPerRow && k < num; k++, j++) {
            double aspectRatio = double(vecMat[k].first.cols) / vecMat[k].first.rows;
            int temp = int(ceil(resizeHeight * aspectRatio));
            resizeWidth.push_back(temp);
            thisRowLen += temp;
        }
        if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
        }
    }

    // Create canvas
    int windowWidth = maxRowLength;
    Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

    // Draw Images
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < num; k++, j++) {
            int x = x_end;
            int y_color = y - edgeThickness / 2;
            int x_color = x - edgeThickness / 2;

            // Determine Highlight
            Rect roi_color(x_color, y_color, resizeWidth[k] + edgeThickness, resizeHeight + edgeThickness);
            Size s_color = canvasImage(roi_color).size();
            Mat target_ROI_color;
            string this_scene = getScene(vecMat[k].second, "");
            if (this_scene != scene) {
                target_ROI_color = Mat (s_color, CV_8UC3, Scalar(0., 0., 255.));
            } else {
                target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 0.));
            }
            target_ROI_color.copyTo(canvasImage(roi_color));

            // Draw Image
            Rect roi(x, y, resizeWidth[k], resizeHeight);
            Size s = canvasImage(roi).size();
            // change the number of channels to three
            Mat target_ROI(s, CV_8UC3);
            if (vecMat[k].first.channels() != canvasImage.channels()) {
                if (vecMat[k].first.channels() == 1) {
                    cvtColor(vecMat[k].first, target_ROI, COLOR_GRAY2BGR);
                }
            } else {
                vecMat[k].first.copyTo(target_ROI);
            }
            resize(target_ROI, target_ROI, s);
            if (target_ROI.type() != canvasImage.type()) {
                target_ROI.convertTo(target_ROI, canvasImage.type());
            }
            target_ROI.copyTo(canvasImage(roi));
            x_end += resizeWidth[k] + edgeThickness;
        }
    }
    imshow(title, canvasImage);
    waitKey();
}

void functions::showSpaced(const string & query_image, const vector<string> & spaced, const string & ext) {
    string scene = getScene(query_image, "");

    vector<pair<Mat, string>> vecMat;
    vecMat.reserve(spaced.size());
    for(const auto & p : spaced) {
        vecMat.emplace_back(imread(p + ext), p);
    }


    // Get window parameters
    int nRows = int(sqrt(spaced.size()));
    int windowHeight = 5000;
    int edgeThickness = 20;
    int imagesPerRow = int(spaced.size() / nRows);
    int resizeHeight = int (floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0))) - edgeThickness;
    int maxRowLength = 0;
    std::vector<int> resizeWidth;
    for (int i = 0; i < nRows; i++) {
        int thisRowLen = 0;
        for (int k = 0; k < imagesPerRow; k++) {
            double aspectRatio = double(vecMat[i].first.cols) / vecMat[i].first.rows;
            int temp = int(ceil(resizeHeight * aspectRatio));
            resizeWidth.push_back(temp);
            thisRowLen += temp;
        }
        if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
        }
    }

    // Create canvas
    int windowWidth = maxRowLength;
    Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

    // Draw Images
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < spaced.size(); k++, j++) {
            int x = x_end;
            int y_color = y - edgeThickness / 2;
            int x_color = x - edgeThickness / 2;

            // Determine Highlight
            Rect roi_color(x_color, y_color, resizeWidth[k] + edgeThickness, resizeHeight + edgeThickness);
            Size s_color = canvasImage(roi_color).size();
            Mat target_ROI_color;
            string this_scene = getScene(vecMat[k].second, "");
            if (this_scene != scene) {
                target_ROI_color = Mat (s_color, CV_8UC3, Scalar(0., 0., 255.));
            } else {
                target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 0.));
            }
            target_ROI_color.copyTo(canvasImage(roi_color));

            // Draw Image
            Rect roi(x, y, resizeWidth[k], resizeHeight);
            Size s = canvasImage(roi).size();
            // change the number of channels to three
            Mat target_ROI(s, CV_8UC3);
            if (vecMat[k].first.channels() != canvasImage.channels()) {
                if (vecMat[k].first.channels() == 1) {
                    cvtColor(vecMat[k].first, target_ROI, COLOR_GRAY2BGR);
                }
            } else {
                vecMat[k].first.copyTo(target_ROI);
            }
            resize(target_ROI, target_ROI, s);
            if (target_ROI.type() != canvasImage.type()) {
                target_ROI.convertTo(target_ROI, canvasImage.type());
            }
            target_ROI.copyTo(canvasImage(roi));
            x_end += resizeWidth[k] + edgeThickness;
        }
    }
    imshow(query_image+" Top K", canvasImage);
    waitKey();
}

void functions::showSpacedInTop(int rows, int cols,
                                const string & query_image, const vector<string> & returned,
                                const vector<string> & spaced, const string & ext) {

    int num = int(returned.size());

    string scene = getScene(query_image, "");

    vector<pair<Mat, string>> vecMat;
    vecMat.reserve(num);
    for (const auto &p: returned) {
        vecMat.emplace_back(imread(p + ext), p);
    }


    // Get window parameters
    int nRows = rows;
    int windowHeight = 5000;
    int edgeThickness = 20;
    int imagesPerRow = cols;
    int resizeHeight = int(floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0))) - edgeThickness;
    int maxRowLength = 0;
    std::vector<int> resizeWidth;
    for (int i = 0; i < nRows; i++) {
        int thisRowLen = 0;
        for (int k = 0; k < imagesPerRow; k++) {
            double aspectRatio = double(vecMat[i].first.cols) / vecMat[i].first.rows;
            int temp = int(ceil(resizeHeight * aspectRatio));
            resizeWidth.push_back(temp);
            thisRowLen += temp;
        }
        if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
        }
    }

    // Create canvas
    int windowWidth = maxRowLength;
    Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

    // Draw Images
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < num; k++, j++) {
            int x = x_end;
            int y_color = y - edgeThickness / 2;
            int x_color = x - edgeThickness / 2;

            // Determine Highlight
            Rect roi_color(x_color, y_color, resizeWidth[k] + edgeThickness, resizeHeight + edgeThickness);
            Size s_color = canvasImage(roi_color).size();
            Mat target_ROI_color;
            string this_scene = getScene(vecMat[k].second, "");
            if (this_scene != scene) {
                target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 255.));
            } else {
                if (count(spaced.begin(), spaced.end(), vecMat[k].second)) {
                    target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 255., 0.));
                } else {
                    target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 0.));
                }
            }
            target_ROI_color.copyTo(canvasImage(roi_color));

            // Draw Image
            Rect roi(x, y, resizeWidth[k], resizeHeight);
            Size s = canvasImage(roi).size();
            // change the number of channels to three
            Mat target_ROI(s, CV_8UC3);
            if (vecMat[k].first.channels() != canvasImage.channels()) {
                if (vecMat[k].first.channels() == 1) {
                    cvtColor(vecMat[k].first, target_ROI, COLOR_GRAY2BGR);
                }
            } else {
                vecMat[k].first.copyTo(target_ROI);
            }
            resize(target_ROI, target_ROI, s);
            if (target_ROI.type() != canvasImage.type()) {
                target_ROI.convertTo(target_ROI, canvasImage.type());
            }
            target_ROI.copyTo(canvasImage(roi));
            x_end += resizeWidth[k] + edgeThickness;
        }
    }
    string title = "K=" + to_string(spaced.size());
    imshow(title, canvasImage);
    waitKey();
}

//void functions::showTop1000(const string & query_image, const string & ext, int max_num, double max_dist, int inliers) {
//
//    // Show Query
//    Mat q = imread(query_image + ext);
//    imshow(query_image, q);
//    waitKey();
//
//    // Get Query Scene
//    string scene = getScene(query_image, "");
//
//    // Get Top 1000 NetVLAD images
//    vector<string> topN = getTopN(query_image, ext, 1000);
//    vector<pair<Mat, string>> vecMat;
//    vecMat.reserve(topN.size());
//    for(const auto & p : topN) {
//        vecMat.emplace_back(imread(p + ext), p);
//    }
//
//    // Threshold, filter, and space the top 1000
//    vector<string> retrieved; vector<double> distances;
//    functions::retrieveSimilar(query_image, "", ext, max_num, max_dist, retrieved, distances);
//    vector<string> spaced = functions::optimizeSpacing(query_image, retrieved, inliers, false, "7-Scenes");
//
//    // Do the following for images/100 windows
//    int N = 100;
//    int num = int (vecMat.size()) / 100;
//    for (int n = 0; n < num; n++) {
//
//        // Get window parameters
//        int nRows = 10;
//        int windowHeight = 5000;
//        nRows = nRows > N ? N : nRows;
//        int edgeThickness = 20;
//        int imagesPerRow = ceil(double(N) / nRows);
//        int resizeHeight = int (floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0))) - edgeThickness;
//        int maxRowLength = 0;
//        std::vector<int> resizeWidth;
//        for (int i = 0; i < N;) {
//            int thisRowLen = 0;
//            for (int k = 0; k < imagesPerRow; k++) {
//                double aspectRatio = double(vecMat[100 * n + i].first.cols) / vecMat[100 * n + i].first.rows;
//                int temp = int(ceil(resizeHeight * aspectRatio));
//                resizeWidth.push_back(temp);
//                thisRowLen += temp;
//                if (++i == N) break;
//            }
//            if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
//                maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
//            }
//        }
//
//        // Create canvas
//        int windowWidth = maxRowLength;
//        Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));
//
//        // Draw Images
//        for (int k = 0, i = 0; i < nRows; i++) {
//            int y = i * resizeHeight + (i + 1) * edgeThickness;
//            int x_end = edgeThickness;
//            for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
//                int x = x_end;
//                int y_color = y - edgeThickness / 2;
//                int x_color = x - edgeThickness / 2;
//
//                // Determine Highlight
//                Rect roi_color(x_color, y_color, resizeWidth[k] + edgeThickness, resizeHeight + edgeThickness);
//                Size s_color = canvasImage(roi_color).size();
//                Mat target_ROI_color;
//                string this_scene = getScene(vecMat[100 * n + k].second, "");
//                if (this_scene != scene) {
//                    target_ROI_color = Mat (s_color, CV_8UC3, Scalar(0., 0., 255.));
//                } else {
//                    if (count(retrieved.begin(), retrieved.end(), vecMat[100 * n + k].second)) {
//                        target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 0.));
//                        if (count(spaced.begin(), spaced.end(), vecMat[100 * n + k].second)) {
//                            target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 255., 0.));
//                        }
//                    } else {
//                        target_ROI_color = Mat(s_color, CV_8UC3, Scalar(255., 255., 255.));
//                    }
//                }
//                target_ROI_color.copyTo(canvasImage(roi_color));
//
//                // Draw Image
//                Rect roi(x, y, resizeWidth[k], resizeHeight);
//                Size s = canvasImage(roi).size();
//                // change the number of channels to three
//                Mat target_ROI(s, CV_8UC3);
//                if (vecMat[100 * n + k].first.channels() != canvasImage.channels()) {
//                    if (vecMat[100 * n + k].first.channels() == 1) {
//                        cvtColor(vecMat[100 * n + k].first, target_ROI, COLOR_GRAY2BGR);
//                    }
//                } else {
//                    vecMat[100 * n + k].first.copyTo(target_ROI);
//                }
//                resize(target_ROI, target_ROI, s);
//                if (target_ROI.type() != canvasImage.type()) {
//                    target_ROI.convertTo(target_ROI, canvasImage.type());
//                }
//                target_ROI.copyTo(canvasImage(roi));
//                x_end += resizeWidth[k] + edgeThickness;
//            }
//        }
//        string title = query_image + "NN from " + to_string(100 * n + 1) + " to " + to_string(100 * n + 100);
//        imshow(title, canvasImage);
//        waitKey();
//    }
//}

Mat functions::projectCentersTo2D(const string & query, const vector<string> & images,
                                  unordered_map<string, cv::Scalar> & seq_colors,
                                  const string & Title) {

    Eigen::Vector3d total {0, 0, 0};
    Eigen::Vector3d z {0, 0, 1};
    vector<tuple<Eigen::Vector3d, Eigen::Vector3d, Scalar>> centers;

    // get query center
    auto c_q = sevenScenes::getT(query);
    total += c_q;

    // get view direction
    auto r_q = sevenScenes::getR(query);
    Eigen::Vector3d dir_q = r_q * z;

    centers.emplace_back(c_q, dir_q, Scalar(0., 0., 0.));

    // do same for all images
    for (const auto & image : images) {
        // get camera center
        auto c = sevenScenes::getT(image);
        total += c;

        // get view direction
        auto r = sevenScenes::getR(image);
        Eigen::Vector3d dir = r * z;

        // get color
        string seq = getSequence(image);
        Scalar color = seq_colors[seq];

        centers.emplace_back(c, dir, color);
    }
    Eigen::Vector3d average = total / centers.size();

    double farthest_x = 0.;
    double farthest_y = 0.;
    double max_z = 0;
    double min_z = 0;
    for (const auto & c : centers) {
        double x_dist = abs(average[0] - get<0>(c)[0]);
        double y_dist = abs(average[2] - get<0>(c)[2]);
        double h = get<0>(c)[1];
        if (x_dist > farthest_x) farthest_x = x_dist;
        if (y_dist > farthest_y) farthest_y = y_dist;
        if (h > max_z) {
            max_z = h;
        } else if (h < min_z) {
            min_z = h;
        }
    }
    double med_z = (max_z + min_z) / 2.;


    double height = 1900.;
    double width = 3000.;
    double avg_radius = 40.;
    double radial_variance = avg_radius * .5;
    double border = 4 * avg_radius;
    double m_over_px_x = (farthest_x / (width/2. - border));
    double m_over_px_y = (farthest_y / (height/2. - border));
    double m_over_px_z = ((max_z - med_z) / radial_variance);
    Mat canvasImage(int (height), int (width), CV_8UC3, Scalar(255., 255., 255.));
    cout << "Window is " << m_over_px_x * width << " meters wide and " << m_over_px_y * height << " meters tall." << endl;


    Point2d im_center {width / 2, height / 2};
    for (const auto & c : centers) {

        // get center point
        Point2d c_pt {
                im_center.x + ((get<0>(c)[0] - average[0]) / m_over_px_x),
                im_center.y + ((get<0>(c)[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (get<0>(c)[1] - med_z) / m_over_px_z;

        // get view point
        double x_dir = get<1>(c)[0] / m_over_px_x;
        double y_dir = get<1>(c)[2] / m_over_px_y;
        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
        double scale =  2. * radius / length_dir;
        x_dir = scale * x_dir;
        y_dir = scale * y_dir;
        Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};

        // get color
        Scalar color = get<2>(c);

        // draw center and direction
        circle(canvasImage, c_pt, int (radius), color, -1);
        line(canvasImage, c_pt, dir_pt, color, int (radius/2.));
    }
    imshow(Title, canvasImage);
    waitKey();
    return canvasImage;
}

cv::Mat functions::showAllImages(const string & query, const vector<string> & images,
                                 unordered_map<string, cv::Scalar> & seq_colors,
                                 const unordered_set<string> & unincluded_all,
                                 const unordered_set<string> & unincluded_top1000,
                                 const string & Title) {

    Eigen::Vector3d total {0, 0, 0};
    Eigen::Vector3d z {0, 0, 1};
    vector<tuple<Eigen::Vector3d, Eigen::Vector3d, Scalar>> centers, first_layer, second_layer;

    // get query center
    auto c_q = sevenScenes::getT(query);
    total += c_q;

    // get view direction
    auto r_q = sevenScenes::getR(query);
    Eigen::Vector3d dir_q = r_q * z;


    // do same for all images
    for (const auto & image : images) {
        // get camera center
        auto c = sevenScenes::getT(image);
        total += c;

        // get view direction
        auto r = sevenScenes::getR(image);
        Eigen::Vector3d dir = r * z;

        // get color
        Scalar color;
        string seq = getSequence(image);
        bool found = false;
        if (unincluded_all.find(image) != unincluded_all.end()) {
            color = cv::Scalar (200., 200., 200.);
            first_layer.emplace_back(c, dir, color);
        } else if (unincluded_top1000.find(image) != unincluded_top1000.end()) {
            color = cv::Scalar (0., 0., 0.);
            first_layer.emplace_back(c, dir, color);
        } else {
            color = seq_colors[seq];
            second_layer.emplace_back(c, dir, color);
        }
        centers.emplace_back(c, dir, color);
    }
    Eigen::Vector3d average = total / centers.size();
    centers.emplace_back(c_q, dir_q, Scalar(0., 0., 0.));
    second_layer.emplace_back(c_q, dir_q, Scalar(0., 0., 0.));

    double farthest_x = 0.;
    double farthest_y = 0.;
    double max_z = 0;
    double min_z = 0;
    for (const auto & c : centers) {
        double x_dist = abs(average[0] - get<0>(c)[0]);
        double y_dist = abs(average[2] - get<0>(c)[2]);
        double h = get<0>(c)[1];
        if (x_dist > farthest_x) farthest_x = x_dist;
        if (y_dist > farthest_y) farthest_y = y_dist;
        if (h > max_z) {
            max_z = h;
        } else if (h < min_z) {
            min_z = h;
        }
    }
    double med_z = (max_z + min_z) / 2.;


    double height = 1900.;
    double width = 3000.;
    double avg_radius = 15.;
    double radial_variance = avg_radius * .5;
    double border = 4 * avg_radius;
    double m_over_px_x = (farthest_x / (width/2. - border));
    double m_over_px_y = (farthest_y / (height/2. - border));
    double m_over_px_z = ((max_z - med_z) / radial_variance);
    Mat canvasImage(int (height), int (width), CV_8UC3, Scalar(255., 255., 255.));
    cout << "Window is " << m_over_px_x * width << " meters wide and " << m_over_px_y * height << " meters tall." << endl;


    Point2d im_center {width / 2, height / 2};
    for (const auto & f : first_layer) {

        // get center point
        Point2d c_pt {
                im_center.x + ((get<0>(f)[0] - average[0]) / m_over_px_x),
                im_center.y + ((get<0>(f)[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (get<0>(f)[1] - med_z) / m_over_px_z;

        // get view point
        double x_dir = get<1>(f)[0] / m_over_px_x;
        double y_dir = get<1>(f)[2] / m_over_px_y;
        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
        double scale =  2. * radius / length_dir;
        x_dir = scale * x_dir;
        y_dir = scale * y_dir;
        Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};

        // get color
        Scalar color = get<2>(f);

        // draw center and direction
        circle(canvasImage, c_pt, int (radius), color, -1);
        line(canvasImage, c_pt, dir_pt, color, int (radius/2.));
    }
    for (const auto & s : second_layer) {

        // get center point
        Point2d c_pt {
                im_center.x + ((get<0>(s)[0] - average[0]) / m_over_px_x),
                im_center.y + ((get<0>(s)[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (get<0>(s)[1] - med_z) / m_over_px_z;

        // get view point
        double x_dir = get<1>(s)[0] / m_over_px_x;
        double y_dir = get<1>(s)[2] / m_over_px_y;
        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
        double scale =  2. * radius / length_dir;
        x_dir = scale * x_dir;
        y_dir = scale * y_dir;
        Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};

        // get color
        Scalar color = get<2>(s);

        // draw center and direction
        circle(canvasImage, c_pt, int (radius), color, -1);
        line(canvasImage, c_pt, dir_pt, color, int (radius/2.));
    }
    imshow(Title, canvasImage);
    waitKey();
    return canvasImage;
}






