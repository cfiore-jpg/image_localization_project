//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/Space.h"
#include "../include/sevenScenes.h"
#include "../include/synthetic.h"
#include "../include/CambridgeLandmarks.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <utility>

using namespace std;

//Point functions
Point * newPoint(const string & name, int num, int num_energies, const Eigen::Vector3d & position) {
    auto * temp = new Point;
    temp->name = name;
    temp->number = num;
    temp->energies = vector<Energy*> (num_energies);
    temp->position = position;
    temp->total_energy = 0.;
    return temp;
}

// Energy Functions
void newEnergy(Point * a, Point * b) {
    auto * temp = new Energy;
    double n = (a->position - b->position).norm();
    if(n == 0) {
        temp->magnitude = DBL_MAX;
    } else {
        temp->magnitude = 1. / n;
    }

    a->energies[b->number] = temp;
    b->energies[a->number] = temp;
    a->total_energy += temp->magnitude;
    b->total_energy += temp->magnitude;
}

//Graph functions
Space::Space() = default;

Space::Space(const vector<string> & images, const string & dataset) {
    if (dataset == "7-Scenes") {
        for (int i = 0; i < images.size(); i++) {
            auto point_to_add = newPoint(images[i], i, int(images.size()), sevenScenes::getT(images[i]));
            for (const auto point: points) {
                newEnergy(point, point_to_add);
            }
            points.push_back(point_to_add);
        }
    } else if (dataset == "Cambridge") {
        for (int i = 0; i < images.size(); i++) {
            Eigen::Vector3d c = - cambridge::getR(images[i]).transpose() * cambridge::getT(images[i]);
            auto point_to_add = newPoint(images[i], i, int(images.size()), c);
            for (const auto point: points) {
                newEnergy(point, point_to_add);
            }
            points.push_back(point_to_add);
        }
    } else if (dataset == "synthetic") {
        for (int i = 0; i < images.size(); i++) {
            auto point_to_add = newPoint(images[i], i, int(images.size()), synthetic::getC(images[i]));
            for (const auto point: points) {
                newEnergy(point, point_to_add);
            }
            points.push_back(point_to_add);
        }
    } else {
        cout << "Unknown Dataset for image retrieval." << endl;
        exit(1);
    }

}

void Space::removeHighestEnergyPoint() {
    sort(points.begin(), points.end(), [](Point * lhs, Point * rhs){
        return lhs->total_energy > rhs->total_energy;
    });
    auto point_to_remove = points[0];
    points.erase(points.begin());
    for (const auto point : points) {
        auto energy = point_to_remove->energies[point->number];
        point->total_energy -= energy->magnitude;
    }
}


void Space::getOptimalSpacing(int N, bool show_process) {
    while (points.size() > N) {
        if (show_process) {
            projectPointsTo2D();
        }
        removeHighestEnergyPoint();
    }
}

vector<string> Space::getPointNames() {
    vector<string> names;
    for (const auto point : points) {
        names.push_back(point->name);
    }
    return names;
}

void Space::projectPointsTo2D () {

    auto highest_energy_point = points[0];
    Eigen::Vector3d total {0, 0, 0};
    for (const auto & point : points) {
        if (point->total_energy > highest_energy_point->total_energy) {
            highest_energy_point = point;
        }
        total += point->position;;
    }
    Eigen::Vector3d average = total / points.size();

    double farthest_x = 0.;
    double farthest_y = 0.;
    double max_z = 0;
    double min_z = 0;
    for (const auto & point : points) {
        double x_dist = abs(average[0] - point->position[0]);
        double y_dist = abs(average[2] - point->position[2]);
        double h = point->position[1];
        if (x_dist > farthest_x) farthest_x = x_dist;
        if (y_dist > farthest_y) farthest_y = y_dist;
        if (h > max_z) {
            max_z = h;
        } else if (h < min_z) {
            min_z = h;
        }
    }
    double med_z = (max_z + min_z) / 2.;


    double height = 2000.; //3.85
    double width = 2000.; //2.39
    double avg_radius = 75.;
    double radial_variance = avg_radius * .5;
    double border = avg_radius;
    double m_over_px_x = (farthest_x / (width/2. - border));
    double m_over_px_y = (farthest_y / (height/2. - border));
    double m_over_px_z = ((max_z - med_z) / radial_variance);
    cv::Mat canvasImage(int (height), int (width), CV_8UC3, cv::Scalar(255., 255., 255.));
    cout << "Window is " << m_over_px_x * width << " meters wide and " << m_over_px_y * height << " meters tall." << endl;


    cv::Point2d im_center {width / 2, height / 2};
    for (const auto & point : points) {

        // get center point
        cv::Point2d c_pt {
                im_center.x + ((point->position[0] - average[0]) / m_over_px_x),
                im_center.y + ((point->position[2] - average[2]) / m_over_px_y)
        };

        // get radius
        double radius = avg_radius + (point->position[1] - med_z) / m_over_px_z;


        // get color
        cv::Scalar color;
        if (point != highest_energy_point) {
            color = cv::Scalar (0., 0., 0.);
        }

        // draw center and direction
        cv::circle(canvasImage, c_pt, int (radius), color, -1);
    }

    cv::Point2d c_pt {
            im_center.x + ((highest_energy_point->position[0] - average[0]) / m_over_px_x),
            im_center.y + ((highest_energy_point->position[2] - average[2]) / m_over_px_y)
    };
    double radius = avg_radius + (highest_energy_point->position[1] - med_z) / m_over_px_z;
    cv::circle(canvasImage, c_pt, int (radius), cv::Scalar (0., 255., 0.), -1);

    imshow("Spacing with " + to_string(points.size()) + " points remaining.", canvasImage);
    cv::waitKey(0);
}

