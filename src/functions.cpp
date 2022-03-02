//
// Created by Cameron Fiore on 12/15/21.
//

#include "../include/imageMatcher_Orb.h"
#include "../include/sevenScenes.h"
#include "../include/functions.h"
#include "../include/Space.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define PI 3.14159265
#define FOLDER "/Users/cameronfiore/C++/image_localization_project/data/"
#define EXT ".color.png"

using namespace std;
using namespace cv;

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
    if ((frac - 1.) > 0. && (frac - 1.) < 0.0000000000000010) frac = 1.;
    double rad = acos(frac);
    return rad * 180.0 / PI;
}

double functions::rotationDifference(const Eigen::Matrix3d & r1, const Eigen::Matrix3d & r2) {

    Eigen::Matrix3d R12 = r1 * r2.transpose();
    double trace = R12.trace();
    if (trace > 3.) trace = 3.;
    double theta = acos((trace-1.)/2.);
    theta = theta * 180. / PI;
    return theta;
}

bool functions::findMatches(const string &db_image, const string &query_image, const string &method, double ratio,
                             vector<Point2d> &pts_db, vector<Point2d> &pts_query) {


    vector<KeyPoint> kp_vec1, kp_vec2;
    Mat desc1, desc2;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + EXT, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + EXT, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray1, Mat(), kp_vec1, desc1);
        orb->detectAndCompute(gray2, Mat(), kp_vec2, desc2);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + EXT, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + EXT, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        surf->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        surf->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();

        Mat image1 = imread(query_image + EXT, 0);
        Mat image2 = imread(db_image + EXT, 0);
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
    } else {
        return true;
    }
}

bool functions::findMatchesSorted(const string &db_image, const string &query_image, const string &method, double ratio,
                                  vector<tuple<Point2d, Point2d, double>> & points) {


    vector<KeyPoint> kp_vec1, kp_vec2;
    Mat desc1, desc2;
    if (method == "ORB") {
        Ptr<ORB> orb = ORB::create();
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + EXT, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + EXT, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        orb->detectAndCompute(gray1, Mat(), kp_vec1, desc1);
        orb->detectAndCompute(gray2, Mat(), kp_vec2, desc2);
    } else if (method == "SURF") {
        Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
        cv::UMat image1, image2, gray1, gray2;
        imread(query_image + EXT, IMREAD_COLOR).copyTo(image1);
        cvtColor(image1, gray1, COLOR_RGB2GRAY);
        imread(db_image + EXT, IMREAD_COLOR).copyTo(image2);
        cvtColor(image2, gray2, COLOR_RGB2GRAY);
        surf->detectAndCompute(image1, Mat(), kp_vec1, desc1);
        surf->detectAndCompute(image2, Mat(), kp_vec2, desc2);
    } else if (method == "SIFT") {
        Ptr<SIFT> sift = SIFT::create();
        Mat image1 = imread(query_image + EXT, 0);
        Mat image2 = imread(db_image + EXT, 0);
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

vector<pair<cv::Point2d, cv::Point2d>> functions::findInliersForFundamental(const Eigen::Matrix3d & F, double threshold,
                                                                 const vector<tuple<cv::Point2d, cv::Point2d, double>> & points) {
    vector<pair<cv::Point2d, cv::Point2d>> inliers;
    for (const auto & tup : points) {
        Eigen::Vector3d pt_q {get<0>(tup).x, get<0>(tup).y, 1.};
        Eigen::Vector3d pt_db {get<1>(tup).x, get<1>(tup).y, 1.};

        Eigen::Vector3d epiline = F * pt_db;

        double error = abs(epiline[0] * pt_q[0] + epiline[1] * pt_q[1] + epiline[2]) /
                       sqrt(epiline[0] * epiline[0] + epiline[1] * epiline[1]);

        if (error <= threshold) {
            inliers.emplace_back(cv::Point2d {get<0>(tup).x, get<0>(tup).y},
                                 cv::Point2d {get<1>(tup).x, get<1>(tup).y});
        }
    }
    return inliers;
}

int functions::getRelativePose(const string & db_image, const string & query_image, const double * K, const string & method,
                               Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
    try {
        vector<Point2d> pts_db, pts_q;
        if (!findMatches(db_image, query_image, method, 0.8, pts_db, pts_q)) return 0;

        Mat im = imread(db_image + EXT);
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

bool functions::getRelativePose(vector<cv::Point2d> & pts_db, vector<cv::Point2d> & pts_q, const double * K,
                               Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
    try {
        Mat mask;
        Mat K_mat = (Mat_<double>(3, 3) <<
                K[0], 0., K[2],
                0., K[1], K[3],
                0., 0.,   1.);

        Mat E_kq = findEssentialMat(pts_db, pts_q, K_mat, RANSAC, 0.999999, 3.0, mask);

        vector<Point2d> inlier_db_points, inlier_q_points;
        for (int i = 0; i < mask.rows; i++) {
            if (mask.at<unsigned char>(i)) {
                inlier_db_points.push_back(pts_db[i]);
                inlier_q_points.push_back(pts_q[i]);
            }
        }

        mask.release();
        Mat R, t;
        recoverPose(E_kq, inlier_db_points, inlier_q_points, K_mat, R, t, mask);

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

int functions::getRelativePose3D(const string &db_image, const string &query_image, const string &method,
                                Eigen::Matrix3d &R_kq, Eigen::Vector3d &t_kq) {
//    try {
    vector<Point2d> pts_db, pts_q;
    if (!findMatches(db_image, query_image, method, 0.8, pts_db, pts_q)) { return 0; }

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

    Mat im = imread(db_image + EXT);
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

        circle(im1, pt1, 5, color, -1);
        circle(im2, pt2, 5, color, -1);
    }
    return total_rep_error / double (pts1.size());
}

vector<string> functions::optimizeSpacing(const vector<string> & images, int N, bool show_process, const string & dataset) {
    Space space (images, dataset);
    space.getOptimalSpacing(N, show_process);
    vector<string> names = space.getPointNames();
    return names;
}

// DATABASE PREPARATION FUNCTIONS
void functions::createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating image vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = int (info.size());
    } else
    {
        begin = scene;
        end = scene + 1;
    }
    for (int i = begin; i < end; ++i)
    {
        string folder = get<0>(info[i]);
        string imageList = get<1>(info[i]);
        vector<string> train = get<2>(info[i]);
        for (auto & seq : train)
        {
            ifstream im_list(imageList);
            if (im_list.is_open())
            {
                string image;
                while (getline(im_list, image))
                {
                    string file = "seq-"; file.append(seq).append("/").append(image);
                    listImage.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}

void functions::createQueryVector(vector<string> &listQuery, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating query vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = int (info.size());
    } else
    {
        begin = scene;
        end = scene + 1;
    }
    for (int i = begin; i < end; ++i)
    {
        string folder = get<0>(info[i]);
        string imageList = get<1>(info[i]);
        vector<string> test = get<3>(info[i]);
        for (auto & seq : test) {
            ifstream im_list(imageList);
            if (im_list.is_open()) {
                string image;
                while (getline(im_list, image)) {
                    string file = "seq-"; file.append(seq).append("/").append(image);
                    listQuery.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}



////SURF functions
//void functions::loadFeaturesSURF(const vector<string> &listImage, vector<vector<vector<float>>> &features)
//{
//    features.clear();
//    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(400, 4, 2, false);
//
//    for (const string& im : listImage)
//    {
//        Mat image = imread(im + EXT, IMREAD_GRAYSCALE);
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
//        Mat image = imread(im + EXT, IMREAD_GRAYSCALE);
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

string functions::getScene(const string & image) {
    string scene = image;
    scene.erase(0, scene.find("data/") + 5);
    scene = scene.substr(0, scene.find("/seq-"));
    return scene;
}

string functions::getSequence(const string & image) {
    string seq = image;
    string scene = functions::getScene(image);
    seq.erase(0, seq.find(scene) + scene.size() + 5);
    seq = seq.substr(0, seq.find("/frame"));
    return seq;
}

vector<string> functions::getTopN(const string& query_image, int N) {
    vector<string> result;
    ifstream file (query_image + EXT + ".1000nn.txt");
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
                fn = im.substr(0, im.find(EXT));
            }
            result.push_back(fn);
            n++;
        }
    }
    return result;
}

vector<string> functions::retrieveSimilar(const string& query_image, int max_num, double max_descriptor_dist) {
    int N = (max_num > 20) ? 20 : max_num;
    vector<string> similar;
    ifstream file(query_image + EXT + ".1000nn.txt");
    if (file.is_open()) {
        string line;
        double descriptor_dist;
        string scene;
        unordered_map<string, int> mc;
        string most_common;
        int num_most_common = 0;
        while (similar.size() < max_num && getline(file, line)) {
            stringstream ss(line);
            string buf, fn;
            bool name = true;
            while (ss >> buf) {
                if (name) {
                    fn = buf.substr(0, buf.find(EXT));
                    name = false;
                } else {
                    descriptor_dist = stod(buf);
                }
            }

            if (descriptor_dist > max_descriptor_dist) break;

            if (scene.empty()) {
                similar.push_back(fn);
                if (similar.size() <= N) {
                    string this_scene = getScene(fn);
                    if (mc.find(this_scene) == mc.end()) {
                        mc[this_scene] = 1;
                    } else {
                        mc[this_scene]++;
                    }
                    if (mc[this_scene] > num_most_common) most_common = this_scene;
                } else {
                    scene = most_common;
                    vector<string> filtered;
                    for (const auto & s: similar) {
                        if (getScene(s) == scene) filtered.push_back(s);
                    }
                    similar = filtered;
                }
            } else {
                if (getScene(fn) == scene) similar.push_back(fn);
            }
        }
    }
    return similar;
}

vector<string> functions::spaceWithMostMatches (const string & query_image, const double * cam_matrix, int K,
                                                int N_thresh, double max_descriptor_dist, double separation, int min_matches,
                                                vector<Eigen::Matrix3d> & R_k,
                                                vector<Eigen::Vector3d> & t_k,
                                                vector<Eigen::Matrix3d> & R_qk,
                                                vector<Eigen::Vector3d> & t_qk) {

    auto images = retrieveSimilar(query_image, N_thresh, max_descriptor_dist);

    vector<string> to_return;
    to_return.push_back(images[0]);

    Eigen::Matrix3d R_1; Eigen::Vector3d t_1;
    sevenScenes::getAbsolutePose(images[0], R_1, t_1);
    R_k.push_back(R_1); t_k.push_back(t_1);

    Eigen::Matrix3d R_q1; Eigen::Vector3d t_q1;
    getRelativePose(images[0], query_image, cam_matrix, "SIFT", R_q1, t_q1);
    R_qk.push_back(R_q1); t_qk.push_back(t_q1);

    int i = 1;
    while (R_k.size() < K && i < images.size()) {

        Eigen::Matrix3d R_i; Eigen::Vector3d t_i;
        sevenScenes::getAbsolutePose(images[i], R_i, t_i);
        Eigen::Vector3d c_i = sevenScenes::getT(images[i]);

        bool too_close = false;
        for (const auto & im : to_return) {
            Eigen::Vector3d c_j = sevenScenes::getT(im);
            double dist = getDistBetween(c_i, c_j);
            if (dist < separation) {
                too_close = true;
                break;
            }
        }

        if (!too_close) {
            Eigen::Matrix3d R_qi; Eigen::Vector3d t_qi;
            int num = getRelativePose(images[i], query_image, cam_matrix, "SIFT", R_qi, t_qi);

            if (num >= min_matches) {
                to_return.push_back(images[i]);
                R_k.push_back(R_i); t_k.push_back(t_i);
                R_qk.push_back(R_qi); t_qk.push_back(t_qi);
            }
        }
        i++;
    }
    return to_return;
}





// Visualize
void functions::showTop1000(const string & query_image,  int max_num, double max_descriptor_dist, int inliers) {

    // Show Query
    Mat q = imread(query_image + EXT);
    imshow(query_image, q);
    waitKey();

    // Get Query Scene
    string scene = getScene(query_image);

    // Get Top 1000 NetVLAD images
    vector<string> topN = getTopN(query_image, 1000);
    vector<pair<Mat, string>> vecMat;
    vecMat.reserve(topN.size());
    for(const auto & p : topN) {
        vecMat.emplace_back(imread(p + EXT), p);
    }

    // Threshold, filter, and space the top 1000
    vector<string> retrieved = functions::retrieveSimilar(query_image, max_num, max_descriptor_dist);
    vector<string> spaced = functions::optimizeSpacing(retrieved, inliers, false, "7-Scenes");

    // Do the following for images/100 windows
    int N = 100;
    int num = int (vecMat.size()) / 100;
    for (int n = 0; n < num; n++) {

        // Get window parameters
        int nRows = 10;
        int windowHeight = 5000;
        nRows = nRows > N ? N : nRows;
        int edgeThickness = 20;
        int imagesPerRow = ceil(double(N) / nRows);
        int resizeHeight = int (floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0))) - edgeThickness;
        int maxRowLength = 0;
        std::vector<int> resizeWidth;
        for (int i = 0; i < N;) {
            int thisRowLen = 0;
            for (int k = 0; k < imagesPerRow; k++) {
                double aspectRatio = double(vecMat[100 * n + i].first.cols) / vecMat[100 * n + i].first.rows;
                int temp = int(ceil(resizeHeight * aspectRatio));
                resizeWidth.push_back(temp);
                thisRowLen += temp;
                if (++i == N) break;
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
            for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
                int x = x_end;
                int y_color = y - edgeThickness / 2;
                int x_color = x - edgeThickness / 2;

                // Determine Highlight
                Rect roi_color(x_color, y_color, resizeWidth[k] + edgeThickness, resizeHeight + edgeThickness);
                Size s_color = canvasImage(roi_color).size();
                Mat target_ROI_color;
                string this_scene = getScene(vecMat[100 * n + k].second);
                if (this_scene != scene) {
                    target_ROI_color = Mat (s_color, CV_8UC3, Scalar(0., 0., 255.));
                } else {
                    if (count(retrieved.begin(), retrieved.end(), vecMat[100 * n + k].second)) {
                        target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 0., 0.));
                        if (count(spaced.begin(), spaced.end(), vecMat[100 * n + k].second)) {
                            target_ROI_color = Mat(s_color, CV_8UC3, Scalar(0., 255., 0.));
                        }
                    } else {
                        target_ROI_color = Mat(s_color, CV_8UC3, Scalar(255., 255., 255.));
                    }
                }
                target_ROI_color.copyTo(canvasImage(roi_color));

                // Draw Image
                Rect roi(x, y, resizeWidth[k], resizeHeight);
                Size s = canvasImage(roi).size();
                // change the number of channels to three
                Mat target_ROI(s, CV_8UC3);
                if (vecMat[100 * n + k].first.channels() != canvasImage.channels()) {
                    if (vecMat[100 * n + k].first.channels() == 1) {
                        cvtColor(vecMat[100 * n + k].first, target_ROI, COLOR_GRAY2BGR);
                    }
                } else {
                    vecMat[100 * n + k].first.copyTo(target_ROI);
                }
                resize(target_ROI, target_ROI, s);
                if (target_ROI.type() != canvasImage.type()) {
                    target_ROI.convertTo(target_ROI, canvasImage.type());
                }
                target_ROI.copyTo(canvasImage(roi));
                x_end += resizeWidth[k] + edgeThickness;
            }
        }
        string title = query_image + "NN from " + to_string(100 * n + 1) + " to " + to_string(100 * n + 100);
        imshow(title, canvasImage);
        waitKey();
    }
}

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

    centers.emplace_back(c_q, dir_q, Scalar(0., 0., 255.));

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
    double avg_radius = 15.;
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

    centers.emplace_back(c_q, dir_q, Scalar(0., 0., 255.));
    second_layer.emplace_back(c_q, dir_q, Scalar(0., 0., 255.));
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






