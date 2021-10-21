//
// Created by Cameron Fiore on 9/30/21.
//

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/rotation.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using namespace std;



// matches_k in the form of q_x,q_y,k_x,k_y
// q_pt * M * k_pt
struct EpipolarConsensusError {
    EpipolarConsensusError(const vector<vector<double>>& cam_k, const vector<vector<vector<double>>>& matches_k)
    : cam_k(cam_k), matches_k(matches_k) {}

    template <typename T>
    bool operator()(const T* const camera, T* residuals) const {

        int residual_num = 0;
        for(int c = 0; c < cam_k.size(); c++) {

            T angle_axis[3];
            angle_axis[0] = camera[0];
            angle_axis[1] = camera[1];
            angle_axis[2] = camera[2];

            T R_wq[9];
            ceres::AngleAxisToRotationMatrix(angle_axis, R_wq);

            T R_kq[9];
            R_kq[0] = T(cam_k[c][0]) * R_wq[0] + T(cam_k[c][1]) * R_wq[1] + T(cam_k[c][2]) * R_wq[2];
            R_kq[1] = T(cam_k[c][3]) * R_wq[0] + T(cam_k[c][4]) * R_wq[1] + T(cam_k[c][5]) * R_wq[2];
            R_kq[2] = T(cam_k[c][6]) * R_wq[0] + T(cam_k[c][7]) * R_wq[1] + T(cam_k[c][8]) * R_wq[2];

            R_kq[3] = T(cam_k[c][0]) * R_wq[3] + T(cam_k[c][1]) * R_wq[4] + T(cam_k[c][2]) * R_wq[5];
            R_kq[4] = T(cam_k[c][3]) * R_wq[3] + T(cam_k[c][4]) * R_wq[4] + T(cam_k[c][5]) * R_wq[5];
            R_kq[5] = T(cam_k[c][6]) * R_wq[3] + T(cam_k[c][7]) * R_wq[4] + T(cam_k[c][8]) * R_wq[5];

            R_kq[6] = T(cam_k[c][0]) * R_wq[6] + T(cam_k[c][1]) * R_wq[7] + T(cam_k[c][2]) * R_wq[8];
            R_kq[7] = T(cam_k[c][3]) * R_wq[6] + T(cam_k[c][4]) * R_wq[7] + T(cam_k[c][5]) * R_wq[8];
            R_kq[8] = T(cam_k[c][6]) * R_wq[6] + T(cam_k[c][7]) * R_wq[7] + T(cam_k[c][8]) * R_wq[8];


            T t_kq[3];
            t_kq[0] = T(cam_k[c][9]) - (R_kq[0] * camera[3] + R_kq[1] * camera[4] * R_kq[2] * camera[5]);
            t_kq[1] = T(cam_k[c][10]) - (R_kq[3] * camera[3] + R_kq[4] * camera[4] * R_kq[5] * camera[5]);
            t_kq[2] = T(cam_k[c][11]) - (R_kq[6] * camera[3] + R_kq[7] * camera[4] * R_kq[8] * camera[5]);


            T E_kq[9];
            E_kq[0] = -R_kq[3] * t_kq[2] + R_kq[6] * t_kq[1];
            E_kq[1] = -R_kq[4] * t_kq[2] + R_kq[7] * t_kq[1];
            E_kq[2] = -R_kq[5] * t_kq[2] + R_kq[8] * t_kq[1];

            E_kq[3] = R_kq[0] * t_kq[2] - R_kq[6] * t_kq[0];
            E_kq[4] = R_kq[1] * t_kq[2] - R_kq[7] * t_kq[0];
            E_kq[5] = R_kq[2] * t_kq[2] - R_kq[8] * t_kq[0];

            E_kq[6] = -R_kq[0] * t_kq[1] + R_kq[3] * t_kq[0];
            E_kq[7] = -R_kq[1] * t_kq[1] + R_kq[4] * t_kq[0];
            E_kq[8] = -R_kq[2] * t_kq[1] + R_kq[5] * t_kq[0];


            T K[9];
            K[0] = camera[6];
            K[3] = 0;
            K[6] = camera[7];

            K[1] = 0;
            K[4] = camera[6];
            K[7] = camera[8];

            K[2] = 0;
            K[5] = 0;
            K[8] = 1;


            T det_K = K[0] * K[4];


            T K_inv[9];
            K_inv[0] = (K[4] * K[8]) / det_K;
            K_inv[1] = 0;
            K_inv[2] = 0;

            K_inv[3] = 0;
            K_inv[4] = (K[0] * K[8]) / det_K;
            K_inv[5] = 0;

            K_inv[6] = (-K[6] * K[4]) / det_K;
            K_inv[7] = -(K[0] * K[7]) / det_K;
            K_inv[8] = (K[0] * K[4]) / det_K;


            T K_inv_T[9];
            K_inv_T[0] = K_inv[0];
            K_inv_T[1] = K_inv[3];
            K_inv_T[2] = K_inv[6];

            K_inv_T[3] = K_inv[1];
            K_inv_T[4] = K_inv[4];
            K_inv_T[5] = K_inv[7];

            K_inv_T[6] = K_inv[2];
            K_inv_T[7] = K_inv[5];
            K_inv_T[8] = K_inv[8];


            T L[9];
            L[0] = K_inv_T[0] * E_kq[0] + K_inv_T[3] * E_kq[1] + K_inv_T[6] * E_kq[2];
            L[1] = K_inv_T[1] * E_kq[0] + K_inv_T[4] * E_kq[1] + K_inv_T[7] * E_kq[2];
            L[2] = K_inv_T[2] * E_kq[0] + K_inv_T[5] * E_kq[1] + K_inv_T[8] * E_kq[2];

            L[3] = K_inv_T[0] * E_kq[3] + K_inv_T[3] * E_kq[4] + K_inv_T[6] * E_kq[5];
            L[4] = K_inv_T[1] * E_kq[3] + K_inv_T[4] * E_kq[4] + K_inv_T[7] * E_kq[5];
            L[5] = K_inv_T[2] * E_kq[3] + K_inv_T[5] * E_kq[4] + K_inv_T[8] * E_kq[5];

            L[6] = K_inv_T[0] * E_kq[6] + K_inv_T[3] * E_kq[7] + K_inv_T[6] * E_kq[8];
            L[7] = K_inv_T[1] * E_kq[6] + K_inv_T[4] * E_kq[7] + K_inv_T[7] * E_kq[8];
            L[8] = K_inv_T[2] * E_kq[6] + K_inv_T[5] * E_kq[7] + K_inv_T[8] * E_kq[8];


            T M[9];
            M[0] = L[0] * K_inv[0] + L[3] * K_inv[1] + L[6] * K_inv[2];
            M[1] = L[1] * K_inv[0] + L[4] * K_inv[1] + L[7] * K_inv[2];
            M[2] = L[2] * K_inv[0] + L[5] * K_inv[1] + L[8] * K_inv[2];

            M[3] = L[0] * K_inv[3] + L[3] * K_inv[4] + L[6] * K_inv[5];
            M[4] = L[1] * K_inv[3] + L[4] * K_inv[4] + L[7] * K_inv[5];
            M[5] = L[2] * K_inv[3] + L[5] * K_inv[4] + L[8] * K_inv[5];

            M[6] = L[0] * K_inv[6] + L[3] * K_inv[7] + L[6] * K_inv[8];
            M[7] = L[1] * K_inv[6] + L[4] * K_inv[7] + L[7] * K_inv[8];
            M[8] = L[2] * K_inv[6] + L[5] * K_inv[7] + L[8] * K_inv[8];


            T l[3];
            for (int i = 0; !matches_k[c].empty(); i++) {
                l[0] = T(matches_k[c][i][0])*M[0] + T(matches_k[c][i][1])*M[1] + M[2];
                l[1] = T(matches_k[c][i][0])*M[3] + T(matches_k[c][i][1])*M[4] + M[5];
                l[2] = T(matches_k[c][i][0])*M[6] + T(matches_k[c][i][1])*M[7] + M[8];

                residuals[residual_num] = l[0]*T(matches_k[c][i][2]) + l[1]*T(matches_k[c][i][3]) + l[2];
                residual_num++;
            }
        }

        return true;
    }


    static ceres::CostFunction* Create(const vector<vector<double>>& cam_k,
                                       const vector<vector<vector<double>>>& matches_k) {
        int num_residuals = 0;
        for (const auto & match : matches_k) {
            num_residuals += match.size();
        }
        return (new ceres::AutoDiffCostFunction<EpipolarConsensusError, num_residuals, 9>(
                new EpipolarConsensusError(cam_k, matches_k)));

    }

    vector<vector<double>> cam_k;
    vector<vector<vector<double>>> matches_k;
};