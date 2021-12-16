//
// Created by Cameron Fiore on 12/14/21.
//

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "../include/sevenScenes.h"
#include <Eigen/Dense>
#include "ceres/rotation.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using namespace std;
void createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene);
int num_matches = 10;
struct SnavelyReprojectionError {
    SnavelyReprojectionError(const vector<double>& E,
                             const vector<vector<double>>& points_i,
                             const vector<vector<double>>& points_j)
            : E(E), points_i(points_i), points_j(points_j) {}

    template <typename T>
    bool operator()(const T* const K, T* residuals) const {
        for (int p = 0; p < points_i.size(); p++) {
            vector<double> point_i = points_i[p];
            vector<double> point_j = points_j[p];

            T first[3];
            first[0] = 1.0/K[0]*(point_i[0] - K[2]);
            first[1] = 1.0/K[1]*(point_i[1] - K[3]);
            first[2] = T(1.0);

            T second[3];
            second[0] = E[0]*first[0] + E[1]*first[1] + E[2]*first[2];
            second[1] = E[3]*first[0] + E[4]*first[1] + E[5]*first[2];
            second[2] = E[6]*first[0] + E[7]*first[1] + E[8]*first[2];

            T third[3];
            third[0] = 1.0/K[0]*second[0];
            third[1] = 1.0/K[1]*second[1];
            third[2] = -K[2]/K[0]*second[0] - K[3]/K[1]*second[1] + second[2];


            T r = ceres::abs(point_j[0]*third[0] + point_j[1]*third[1] + third[2]);
            T denom = ceres::sqrt(third[0]*third[0] + third[1]*third[1]);
            T d = r/denom;
            residuals[p] = d;
        }
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const vector<double>& E,
                                       const vector<vector<double>>& points_i,
                                       const vector<vector<double>>& points_j) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, num_matches, 4>(
                new SnavelyReprojectionError(E, points_i, points_j)));
    }

    vector<double> E;
    vector<vector<double>> points_i;
    vector<vector<double>> points_j;
};

int main()
{
    vector<string> images;
    vector<tuple<string, string, vector<string>, vector<string>>> info = sevenScenes::createInfoVector();
    createImageVector(images, info, 0);
    Problem problem;
    double K[4];
    K[0] = 585;
    K[1] = 585;
    K[2] = 320;
    K[3] = 240;

    for (int i = 0; i < 3999; i+=1) {
        string im_i = images[i];
        Eigen::Matrix3d R_iw = sevenScenes::getR(im_i);
        Eigen::Vector3d t_iw = sevenScenes::getT(im_i);
        Eigen::Matrix3d R_wi = R_iw.transpose();
        Eigen::Vector3d t_wi = -R_wi * t_iw;

        string im_j = images[i+1];
        Eigen::Matrix3d R_jw = sevenScenes::getR(im_j);
        Eigen::Vector3d t_jw = sevenScenes::getT(im_j);
        Eigen::Matrix3d R_wj = R_jw.transpose();
        Eigen::Vector3d t_wj = -R_wj * t_jw;

        Eigen::Matrix3d R_ij = R_wi.transpose() * R_wj;
        Eigen::Vector3d t_ij = t_wj - R_ij * t_wi;

        Eigen::Matrix3d t_cross;
        t_cross << 0, -t_ij(2), t_ij(1),
                t_ij(2), 0, -t_ij(0),
                -t_ij(1), t_ij(0), 0;
        Eigen::Matrix3d E = t_cross * R_ij;

        vector<double> E_vec{E(0, 0),
                             E(0, 1),
                             E(0, 2),
                             E(1, 0),
                             E(1, 1),
                             E(1, 2),
                             E(2, 0),
                             E(2, 1),
                             E(2, 2)};

        vector<cv::Point2d> pts_i, pts_j;
        sevenScenes::findMatches(im_i, im_j, "SIFT", pts_i, pts_j);

        vector<vector<double>> points_i, points_j;
        int idx = 0;
        while (idx < pts_i.size() && idx < num_matches) {
            points_i.push_back(vector<double> {pts_i[idx].x, pts_i[idx].y, 1});
            points_j.push_back(vector<double> {pts_j[idx].x, pts_j[idx].y, 1});
            idx++;
        }
        if (points_i.size() < num_matches) {
            continue;
        }

        CostFunction *cost_function =
                SnavelyReprojectionError::Create(E_vec, points_i, points_j);
        ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function,
                                 loss_function,
                                 K);
    }
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    cout << K[0] << ", " << K[1] << ", " << K[2] << ", " <<K[3] << endl;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    cout << K[0] << ", " << K[1] << ", " << K[2] << ", " <<K[3] << endl;
}

void createImageVector(vector<string> &listImage, vector<tuple<string, string, vector<string>, vector<string>>> &info, int scene)
{
    cout << "Creating image vector..." << endl;
    int begin;
    int end;
    if (scene == -1)
    {
        begin = 0;
        end = info.size();
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
                    string file = "seq-" + seq + "/" + image;
                    listImage.push_back(folder + file);
                }
                im_list.close();
            }
        }
    }
    cout << "done" << endl;
}
