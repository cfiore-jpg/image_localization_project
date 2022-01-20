//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/Space.h"
#include "../include/sevenScenes.h"
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <utility>

using namespace std;

//Node function
Point * newPoint(const string & name, int i, int num_energies, const Eigen::Vector3d & position) {
    auto * temp = new Point;
    temp->name = name;
    temp->number = i;
    temp->energies = vector<Energy*> (num_energies);
    temp->position = position;
    temp->total_energy = 0.;
    return temp;
}

// Edge Functions
void newEnergy(Point * a, Point * b) {
    auto * temp = new Energy;
    temp->magnitude = 1. / pow((a->position - b->position).norm(), 2);

    a->energies[b->number] = temp;
    b->energies[a->number] = temp;
    a->total_energy += temp->magnitude;
    b->total_energy += temp->magnitude;
}

//Graph functions
Space::Space() = default;

Space::Space(const vector<string> & images) {
    for (int i = 0; i < images.size(); i++) {
        auto point_to_add = newPoint(images[i], i, int (images.size()), sevenScenes::getT(images[i]));
        for (const auto point : points) {
            newEnergy(point, point_to_add);
        }
        points.push_back(point_to_add);
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


void Space::getOptimalSpacing(int N) {
    while (points.size() > N) {
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

//void Space::projectPointsTo2D () {
//
//    sort(points.begin(), points.end(), [](Point * lhs, Point * rhs){
//        return lhs->total_energy > rhs->total_energy;
//    });
//
//    // get average 2d position
//    Eigen::Vector3d total {0, 0, 0};
//    for (const auto & point : points) {
//        total += point->position;
//    }
//    Eigen::Vector3d average = total / points.size();
//
//    double farthest_x = 0.;
//    double farthest_y = 0.;
//    double farthest_z = 0.;
//    for (const auto & point : points) {
//        double x_dist = abs(average[0] - point->position[0]);
//        double y_dist = abs(average[2] - point->position[2]);
//        double z_dist = abs(average[1] - point->position[1]);
//        if (x_dist > farthest_x) farthest_x = x_dist;
//        if (y_dist > farthest_y) farthest_y = y_dist;
//        if (z_dist > farthest_z) farthest_z = z_dist;
//
//    }
//
//
//    double height = 2500.;
//    double width = 2500.;
//    double border = 25.;
//    double m_over_px_x = (farthest_x / (width/2. - border));
//    double m_over_px_y = (farthest_y / (height/2. - border)) ;
//    cv::Mat canvasImage(int (height), int (width), CV_8UC3, cv::Scalar(255., 255., 255.));
//
//
//    cv::Point2d im_center {width / 2, height / 2};
//    for (const auto & point : points) {
//
//        // get center point
//        cv::Point2d c_pt {
//                im_center.x + ((get<0>(c)[0] - average[0]) / m_over_px_x),
//                im_center.y + ((get<0>(c)[2] - average[2]) / m_over_px_y)
//        };
//
//        // get view point
//        double x_dir = get<1>(c)[0] / m_over_px_x;
//        double y_dir = get<1>(c)[2] / m_over_px_y;
//        double length_dir = sqrt(pow(x_dir, 2) + pow(y_dir, 2));
//        double scale =  20. / length_dir;
//        x_dir = scale * x_dir;
//        y_dir = scale * y_dir;
//        cv::Point2d dir_pt {c_pt.x + x_dir, c_pt.y + y_dir};
//
//        // get color
//        cv::Scalar color = get<2>(c);
//
//        // draw center and direction
//        cv::circle(canvasImage, c_pt, 10, color, -1);
//        cv::line(canvasImage, c_pt, dir_pt, color, 5.);
//    }
//    cv::imshow(Title, canvasImage);
//    cv::waitKey();
//}

