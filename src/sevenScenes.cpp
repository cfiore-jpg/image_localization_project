//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/sevenScenes.h"

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <base/database.h>
#include <base/image.h>
#include <fstream>

#define PI 3.14159265

using namespace std;

//Pre-processing

vector<tuple<string, string, vector<string>, vector<string>>> sevenScenes::createInfoVector()
{
    vector<tuple<string, string, vector<string>, vector<string>>> info;

    tuple<string, string, vector<string>, vector<string>> chess;
    vector<string> train0 {"01", "02", "04", "06"};
    vector<string> test0 {"03", "05"};
    get<0>(chess) = "/Users/cameronfiore/C++/ImageMatcherProject/data/chess/";
    get<1>(chess) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt";
    get<2>(chess) = train0;
    get<3>(chess) = test0;
    info.push_back(chess);

    tuple<string, string, vector<string>, vector<string>> fire;
    vector<string> train1 {"01", "02"};
    vector<string> test1 {"03", "04"};
    get<0>(fire) = "/Users/cameronfiore/C++/ImageMatcherProject/data/fire/";
    get<1>(fire) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt";
    get<2>(fire) = train1;
    get<3>(fire) = test1;
    info.push_back(fire);

    tuple<string, string, vector<string>, vector<string>> heads;
    vector<string> train2 {"02"};
    vector<string> test2 {"01"};
    get<0>(heads) = "/Users/cameronfiore/C++/ImageMatcherProject/data/heads/";
    get<1>(heads) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt";
    get<2>(heads) = train2;
    get<3>(heads) = test2;
    info.push_back(heads);

    tuple<string, string, vector<string>, vector<string>> office;
    vector<string> train3 {"01", "03", "04", "05", "08", "10"};
    vector<string> test3 {"02", "06", "07", "09"};
    get<0>(office) = "/Users/cameronfiore/C++/ImageMatcherProject/data/office/";
    get<1>(office) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt";
    get<2>(office) = train3;
    get<3>(office) = test3;
    info.push_back(office);

    tuple<string, string, vector<string>, vector<string>> pumpkin;
    vector<string> train4 {"02", "03", "06", "08"};
    vector<string> test4 {"01", "07"};
    get<0>(pumpkin) = "/Users/cameronfiore/C++/ImageMatcherProject/data/pumpkin/";
    get<1>(pumpkin) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt";
    get<2>(pumpkin) = train4;
    get<3>(pumpkin) = test4;
    info.push_back(pumpkin);

    tuple<string, string, vector<string>, vector<string>> redkitchen;
    vector<string> train5 {"01", "02", "05", "07", "08", "11", "13"};
    vector<string> test5 {"03", "04", "06", "12", "14"};
    get<0>(redkitchen) = "/Users/cameronfiore/C++/ImageMatcherProject/data/redkitchen/";
    get<1>(redkitchen) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_1000.txt";
    get<2>(redkitchen) = train5;
    get<3>(redkitchen) = test5;
    info.push_back(redkitchen);

    tuple<string, string, vector<string>, vector<string>> stairs;
    vector<string> train6 {"02", "03", "05", "06"};
    vector<string> test6 {"01", "04"};
    get<0>(stairs) = "/Users/cameronfiore/C++/ImageMatcherProject/data/stairs/";
    get<1>(stairs) = "/Users/cameronfiore/C++/ImageMatcherProject/data/images_500.txt";
    get<2>(stairs) = train6;
    get<3>(stairs) = test6;
    info.push_back(stairs);

    return info;
}



//Math

Eigen::Matrix3d sevenScenes::getR(const string& image)
{
    Eigen::Matrix3d R;

    string poseFile = image + ".pose.txt";
    ifstream im_pose (poseFile);
    if (im_pose.is_open())
    {
        string line;
        int row = 0;
        while (getline(im_pose, line))
        {
            if (row < 3)
            {
                stringstream ss(line);
                string num;
                int col = 0;
                while (ss >> num) {
                    if (col < 3)
                    {
                        R(row, col) = stod(num);
                    }
                    ++col;
                }
                ++row;
            }
        }
        im_pose.close();
        return R;
    } else
    {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Matrix3d l;
        return l;
    }
}

Eigen::Vector3d sevenScenes::getT(const string& image)
{
    Eigen::Vector3d t;

    string poseFile = image + ".pose.txt";
    ifstream im_pose (poseFile);
    if (im_pose.is_open())
    {
        string line;
        int row = 0;
        while (getline(im_pose, line))
        {
            if (row < 3)
            {
                stringstream ss(line);
                string num;
                int col = 0;
                while (ss >> num) {
                    if (col == 3)
                    {
                        t(row) = stod(num);
                    }
                    ++col;
                }
                ++row;
            }
        }
        im_pose.close();
        return t;
    } else
    {
        cout << "Pose file does not exist for: " << image << endl;
        Eigen::Vector3d l;
        return l;
    }
}

double sevenScenes::triangulateRays(const Eigen::Vector3d& ci, const Eigen::Vector3d& di, const Eigen::Vector3d& cj, const Eigen::Vector3d& dj, Eigen::Vector3d &intersect)
{
    Eigen::Vector3d dq = di.cross(dj);
    Eigen::Matrix3d D;
    D.col(0) = di;
    D.col(1) = -dj;
    D.col(2) = dq;
    Eigen::Vector3d b = cj - ci;
    Eigen::Vector3d sol = D.colPivHouseholderQr().solve(b);
    intersect = ci + sol(0)*di + (sol(2)/2)*dq;
    Eigen::Vector3d toReturn = sol(2)*dq;
    return toReturn.norm();
}

double sevenScenes::getDistBetween(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2)
{
    double x = p1(0) - p2(0);
    double y = p1(1) - p2(1);
    double z = p1(2) - p2(2);
    return sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
}

double sevenScenes::getAngleBetween(const Eigen::Vector3d &d1, const Eigen::Vector3d &d2)
{
    return acos(d1.dot(d2) / (d1.norm() * d2.norm())) * 180.0 / PI;
}





// Processing

bool sevenScenes::findMatches(const string& db_image, const string& query_image, const string& method, vector<cv::Point2d>& pts_db, vector<cv::Point2d>& pts_query) {

    cv::Mat image1 = cv::imread(query_image + ".color.png");
    cv::Mat image2 = cv::imread(db_image + ".color.png");

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, false);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    cv::Mat mask1, mask2;
    vector<cv::KeyPoint> kp_vec1, kp_vec2;
    cv::Mat desc1, desc2;

    if(method == "ORB")
    {
        orb->detectAndCompute(image1, mask1, kp_vec1, desc1);
        orb->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else if(method == "SURF")
    {
        surf->detectAndCompute(image1, mask1, kp_vec1, desc1);
        surf->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else if(method == "SIFT")
    {
        sift->detectAndCompute(image1, mask1, kp_vec1, desc1);
        sift->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else
    {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    cv::BFMatcher matcher(cv::NORM_L2, false);
    vector<vector<cv::DMatch>> matches_2nn_12, matches_2nn_21;
    matcher.knnMatch(desc1, desc2, matches_2nn_12, 2);
    matcher.knnMatch(desc2, desc1, matches_2nn_21, 2);
    const double ratio = 0.8;
    vector<cv::Point2d> pts1, pts2;
    for (int i = 0; i < matches_2nn_12.size(); i++) {
        if( matches_2nn_12[i][0].distance/matches_2nn_12[i][1].distance < ratio
            and
            matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].distance
            / matches_2nn_21[matches_2nn_12[i][0].trainIdx][1].distance < ratio )
        {
            if(matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].trainIdx
               == matches_2nn_12[i][0].queryIdx)
            {
                pts1.push_back(kp_vec1[matches_2nn_12[i][0].queryIdx].pt);
                pts2.push_back(kp_vec2[matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].queryIdx].pt);
            }
        }
    }

    if(pts1.size() < 5)
    {
        return false;
    }

    pts_query = pts1;
    pts_db = pts2;
    return true;
}

vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<cv::Point2d>, vector<cv::Point2d>>> sevenScenes::getTopXImageMatches(const vector<pair<int, float>> &result, const vector<string> &listImage, const string &queryImage, int num_images, const string& method) {

    vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<cv::Point2d>, vector<cv::Point2d>>> to_return;
    int i = 0;

    while(i < result.size() and to_return.size() < num_images) {

        string cur_image = listImage[result[i].first];
        Eigen::Matrix3d R = getR(cur_image).transpose();
        Eigen::Vector3d t = -R * getT(cur_image);

        vector<cv::Point2d> pts_db, pts_query;
        if(findMatches(cur_image, queryImage, method, pts_db, pts_query)) {
            to_return.emplace_back(R, t, pts_query, pts_db);
        } else {
            continue;
        }
        i++;
    }

    sort(to_return.begin(), to_return.end(), [](const auto& lhs, const auto& rhs){
        return get<2>(lhs).size() > get<2>(rhs).size();
    });


    return to_return;
}

void sevenScenes::handle(const vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, vector<cv::Point2d>, vector<cv::Point2d>>>& lst, const cv::Mat& K, Eigen::Matrix3d& R, Eigen::Vector3d& t, vector<vector<double>>& cams, vector<vector<vector<double>>>& matches) {

    auto image_j = lst[0];
    auto R_wj = get<0>(image_j);
    auto t_wj = get<1>(image_j);
    auto pts_q_j = get<2>(image_j);
    auto pts_j = get<3>(image_j);

    auto image_k = lst[1];
    auto R_wk = get<0>(image_k);
    auto t_wk = get<1>(image_k);
    auto pts_q_k = get<2>(image_k);
    auto pts_k = get<3>(image_k);

    cv::Mat mask_j, mask_k;
    cv::Mat E_jq = cv::findEssentialMat(pts_j, pts_q_j, K, cv::RANSAC, 0.999, 1.0, mask_j);
    cv::Mat E_kq = cv::findEssentialMat(pts_k, pts_q_k, K, cv::RANSAC, 0.999, 1.0, mask_k);


    vector<cv::Point2d> rp_q_j, rp_j, rp_q_k, rp_k;
    for(int i = 0; i < mask_j.rows; i++) {
        if(mask_j.at<unsigned char>(i)){
            rp_q_j.push_back(pts_q_j[i]);
            rp_j.push_back(pts_j[i]);
        }
    }
    for(int i = 0; i < mask_k.rows; i++) {
        if(mask_k.at<unsigned char>(i)){
            rp_q_k.push_back(pts_q_k[i]);
            rp_k.push_back(pts_k[i]);
        }
    }


    Eigen::Matrix3d R_jq, R_kq;
    Eigen::Vector3d t_jq, t_kq;

    cv::Mat R_cv_j, t_cv_j, maskR_j;
    cv::recoverPose(E_jq, rp_j, rp_q_j,K, R_cv_j, t_cv_j, maskR_j);
    cv::cv2eigen(R_cv_j, R_jq);
    cv::cv2eigen(t_cv_j, t_jq);

    cv::Mat R_cv_k, t_cv_k, maskR_k;
    cv::recoverPose(E_kq, rp_k, rp_q_k, K, R_cv_k, t_cv_k, maskR_k);
    cv::cv2eigen(R_cv_k, R_kq);
    cv::cv2eigen(t_cv_k, t_kq);

    Eigen::Vector3d c_j = -R_wj.transpose()*t_wj;
    Eigen::Vector3d ray_j = R_wj.transpose()*R_jq.transpose()*t_jq;

    Eigen::Vector3d c_k = -R_wk.transpose()*t_wk;
    Eigen::Vector3d ray_k = R_wk.transpose()*R_kq.transpose()*t_kq;

    Eigen::Vector3d c_q_w;
    triangulateRays(c_j, ray_j, c_k, ray_k, c_q_w);



    // This is the initial estimate of pose
    R = R_wj*R_jq;
    t = -R.transpose()*c_q_w;



    // This is the refinement set
    cams.clear();
    matches.clear();
    for(int i = 2; i < lst.size(); i++){
        auto R_wi = get<0>(lst[i]);
        auto t_wi = get<1>(lst[i]);
        vector<cv::Point2d> pts_q = get<2>(lst[i]);
        vector<cv::Point2d> pts_i = get<3>(lst[i]);

        vector<double> cam_i {R_wi(0,0),
                              R_wi(1,0),
                              R_wi(2,0),

                              R_wi(0,1),
                              R_wi(1,1),
                              R_wi(2,1),

                              R_wi(0,2),
                              R_wi(1,2),
                              R_wi(2,2),

                              t_wi[0],
                              t_wi[1],
                              t_wi[2]};
        cams.push_back(cam_i);

        vector<vector<double>> matches_i;
        for(int j = 0; j < pts_q.size(); j++) {
            vector<double> match {pts_q[j].x, pts_q[j].y, pts_i[j].x, pts_i[j].y};
            matches_i.push_back(match);
        }
        matches.push_back(matches_i);
    }
}




