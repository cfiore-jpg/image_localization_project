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

vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> sevenScenes::pairSelection(const vector<pair<int, float>> &result, const vector<string> &listImage, const string &queryImage, double loThresh, int maxNum, int method)
{
    int i = 0;
    vector<tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int>> rel_images;
    while(i < result.size() && rel_images.size() < maxNum)
    {
        string cur_image = listImage[result[i].first];
        Eigen::Vector3d curLoc = getT(cur_image);
        Eigen::Matrix3d Riq;
        Eigen::Vector3d tiq;
        int points;
        if(rel_images.empty())
        {
            if(findEssentialMatrix(cur_image, queryImage, method, Riq, tiq, points))
            {
                rel_images.push_back(tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int> (getR(cur_image), curLoc, Riq, tiq, points));
            }
        } else
        {
            bool tooClose = false;
            for (auto &ri : rel_images)
            {
                double dist = getDistBetween(curLoc, get<1>(ri));
                if (dist < loThresh) {
                    tooClose = true;
                    break;
                }
            }
            if(!tooClose)
            {
                if(findEssentialMatrix(cur_image, queryImage, method, Riq, tiq, points))
                {
                    rel_images.push_back(tuple<Eigen::Matrix3d, Eigen::Vector3d, Eigen::Matrix3d, Eigen::Vector3d, int> (getR(cur_image), curLoc, Riq, tiq, points));
                }
            }
        }
        ++i;
    }
    return rel_images;
}



//Vector Measurements

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

bool sevenScenes::findEssentialMatrix(const string &db_image, const string &query_image, int method, Eigen::Matrix3d &R, Eigen::Vector3d &t, int& num_points)
{

    cv::Mat image1 = cv::imread(query_image + ".color.png");
    cv::Mat image2 = cv::imread(db_image + ".color.png");

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400, 4, 2, false);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    cv::Mat mask1, mask2;
    vector<cv::KeyPoint> kp_vec1, kp_vec2;
    cv::Mat desc1, desc2;

    if(method == 0)
    {
        orb->detectAndCompute(image1, mask1, kp_vec1, desc1);
        orb->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else if(method == 1)
    {
        surf->detectAndCompute(image1, mask1, kp_vec1, desc1);
        surf->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else if(method == 2)
    {
        sift->detectAndCompute(image1, mask1, kp_vec1, desc1);
        sift->detectAndCompute(image2, mask2, kp_vec2, desc2);
    } else
    {
        cout << "Not a valid method for feature matching..." << endl;
        exit(1);
    }

    cv::BFMatcher matcher(cv::NORM_L2, false);
    vector< vector<cv::DMatch> > matches_2nn_12, matches_2nn_21;
    matcher.knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher.knnMatch( desc2, desc1, matches_2nn_21, 2 );
    const double ratio = 0.8;
    vector<cv::Point2d> pts1, pts2;
    for (int i = 0; i < matches_2nn_12.size(); i++) { // i is queryIdx
        if( matches_2nn_12[i][0].distance/matches_2nn_12[i][1].distance < ratio
            and
            matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].distance
            / matches_2nn_21[matches_2nn_12[i][0].trainIdx][1].distance < ratio )
        {
            if(matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].trainIdx
               == matches_2nn_12[i][0].queryIdx)
            {
                pts1.push_back(kp_vec1[matches_2nn_12[i][0].queryIdx].pt);
                pts2.push_back(
                        kp_vec2[matches_2nn_21[matches_2nn_12[i][0].trainIdx][0].queryIdx].pt
                );
            }
        }
    }

    if(pts1.size() < 3)
    {
        return false;
    }

    try {

//        cv::Mat im2_points = image2.clone();
//        for(const auto &p: pts2)
//        {
//            cv::circle(im2_points, p, 3, cv::Scalar(0, 0, 0), -1);
//        }
//        cv::imshow("DB Image Points", im2_points);
//        cv::waitKey();


//        cv::Mat src1;
//        cv::hconcat(image1, image2, src1);
//        for(int i = 0; i < pts1.size(); i++) {
//            cv::line( src1, pts1[i],
//                      cv::Point2d(pts2[i].x + image1.cols, pts2[i].y),
//                      1, 1, 0 );
//        }
//        cv::imshow("Before Filtering", src1);
//        cv::waitKey();
//
//        int num_clusters = pts2.size() / 10;
//        if(num_clusters == 0)
//        {
//            num_clusters = 1;
//        }
//
//        vector<pair<vector<cv::Point2d>, vector<cv::Point2d>>> clusters = sevenScenes::findClusters(pts1, pts2, num_clusters, image2);
//
//        cv::Mat mask, src2;
//        vector<cv::Point2d> inlier_points1, inlier_points2;
//        int color = 0;
//        for(const auto & c:clusters)
//        {
//            src2.release();
//            cv::hconcat(image1, image2, src2);
//            int r = 30*color % 255;
//            int g = 60*color % 255;
//            int b = 90*color % 255;
//            for(int i = 0; i < c.first.size(); i++) {
//                cv::line( src2, c.first[i],
//                          cv::Point2d(c.second[i].x + image1.cols, c.second[i].y),
//                          cv::Scalar (r, g, b), 1, 0 );
//            }
//            cv::imshow("Before Homography", src2);
//            cv::waitKey();
//            try {
//                cv::findHomography(c.first, c.second, cv::RANSAC, 1.0, mask);
//                vector<cv::Point2d> new_points1, new_points2;
//                for (int i = 0; i < mask.rows; i++) {
//                    if (mask.at<unsigned char>(i)) {
//                        inlier_points1.push_back(c.first[i]);
//                        inlier_points2.push_back(c.second[i]);
//                        new_points1.push_back(c.first[i]);
//                        new_points2.push_back(c.second[i]);
//                    }
//                }
//                src2.release();
//                cv::hconcat(image1, image2, src2);
//                for(int i = 0; i < new_points1.size(); i++) {
//                    cv::line( src2, new_points1[i],
//                              cv::Point2d(new_points2[i].x + image1.cols, new_points2[i].y),
//                              cv::Scalar (r, g, b), 1, 0 );
//                }
//                cv::imshow("After Homography", src2);
//                cv::waitKey();
//            } catch (exception &e){}
//            mask.release();
//            ++color;
//        }
//
//        cv::Mat src3;
//        cv::hconcat(image1, image2, src3);
//        for(int i = 0; i < inlier_points1.size(); i++) {
//            cv::line( src3, inlier_points1[i],
//                      cv::Point2d(inlier_points2[i].x + image1.cols, inlier_points2[i].y),
//                      1, 1, 0 );
//        }
//        cv::imshow("After Filtering", src3);
//        cv::waitKey();


//        cv::Mat maskF;
//        cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 2.0, 0.99, maskF);
//
//        vector<cv::Point2d> new_pts1, new_pts2;
//        for(int i = 0; i < maskF.rows; i++) {
//            if(maskF.at<unsigned char>(i)){
//                new_pts1.push_back(pts1[i]);
//                new_pts2.push_back(pts2[i]);
//            }
//        }
//
//        vector<cv::Point2d> inlier_points1, inlier_points2;
//        cv::correctMatches(F, new_pts1, new_pts2, inlier_points1, inlier_points2);

        cv::Mat maskE;
        cv::Mat K = (cv::Mat_<double>(3,3) <<
                500,   0, image1.cols / 2,
                0, 500, image1.rows / 2,
                0.f,   0,   1);
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, maskE);

        vector<cv::Point2d> recover_points1, recover_points2;
        for(int i = 0; i < maskE.rows; i++) {
            if(maskE.at<unsigned char>(i)){
                recover_points1.push_back(pts1[i]);
                recover_points2.push_back(pts2[i]);
            }
        }


        for(int i = 0; i < recover_points1.size(); i++)
        {
            cv::Mat x = (cv::Mat_<double>(3,1) << recover_points1[i].x, recover_points1[i].y, 1);
            cv::Mat x_h = (cv::Mat_<double>(1,3) << recover_points2[i].x, recover_points2[i].y, 1);

            cv::Mat final = x_h * K.inv().t() * E * K.inv() * x;
            cout << final << endl;
//            Eigen::Vector3d value = first.transpose() * x_h;
        }



//        cv::Mat src2;
//        cv::hconcat(image1, image2, src2);
//        for(int i = 0; i < recover_points1.size(); i++) {
//            cv::line( src2, recover_points1[i],
//                      cv::Point2d(recover_points2[i].x + image1.cols, recover_points2[i].y),
//                      1, 1, 0 );
//        }
//        cv::imshow("After RANSAC Filtering", src2);
//        cv::waitKey();

        cv::Mat R_cv, t_cv, maskR;
        cv::recoverPose(E.clone(), recover_points1, recover_points2, K, R_cv, t_cv,maskR);

        num_points = 0;
        for(int i = 0; i < maskR.rows; i++) {
            if(maskE.at<unsigned char>(i)){
                num_points++;
            }
        }

        cv::Mat R_inv = R_cv.t();
        cv::Mat T = -R_inv * t_cv;
        cv::cv2eigen(R_inv, R);
        cv::cv2eigen(T, t);
        return true;
    } catch (exception& e)
    {
        return false;
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




// Hypothesis




