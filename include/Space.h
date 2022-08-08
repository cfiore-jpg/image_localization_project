//
// Created by Cameron Fiore on 7/2/21.
//

#ifndef IMAGEMATCHERPROJECT_ORB_KDTREE_H
#define IMAGEMATCHERPROJECT_ORB_KDTREE_H

#endif //IMAGEMATCHERPROJECT_ORB_KDTREE_H

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>
#include "sevenScenes.h"

using namespace std;

struct Point;
struct Energy;

struct Point
{
    string name {};
    int number {};
    Eigen::Vector3d position;
    vector<Energy*> energies;
    double total_energy {};
};

struct Energy {
    double magnitude {};
};

class Space
{
    Eigen::Vector3d query;
    vector<Point*> points;
public:
    Space ();
    explicit Space (const string & query,
                    const vector<string> & images,
                    const string & dataset);
    void removeHighestEnergyPoint();
    void getOptimalSpacing(int N, bool show_process);
    vector<string> getPointNames();
    void projectPointsTo2D (double & avg_width, double & avg_length,
                            double & m_over_px_x, double & m_over_px_y, double & m_over_px_z,
                            double & max_z, double & min_z);
};

