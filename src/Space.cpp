//
// Created by Cameron Fiore on 7/2/21.
//

#include "../include/Space.h"
#include "../include/aachen.h"
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
Point * newPoint(const string & name,
                 int num,
                 int num_energies,
                 const Eigen::Vector3d & position) {
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
        temp->magnitude = DBL_MAX/2;
    } else {
        temp->magnitude = 1. / pow(n, 2.);
    }

    a->energies[b->number] = temp;
    b->energies[a->number] = temp;
    a->total_energy += temp->magnitude;
    b->total_energy += temp->magnitude;
}

//Graph functions
Space::Space() = default;

Space::Space(const string & query,
             const vector<string> & images,
             const string & dataset) {

    if (dataset == "7-Scenes") {
        for (int i = 0; i < images.size(); i++) {
            auto point_to_add = newPoint(images[i], i, int(images.size()), sevenScenes::getT(images[i]));
            for (const auto point: points) {
                newEnergy(point, point_to_add);
            }
            points.push_back(point_to_add);
        }
        this->query = sevenScenes::getT(query);
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
    } else if (dataset == "aachen") {
        for (int i = 0; i < images.size(); i++) {
            Eigen::Vector3d c;
            if (!aachen::getC(images[i], c)) {
                cout << "Aachen error." << endl;
                exit(1);
            }
            auto point_to_add = newPoint(images[i], i, int(images.size()), c);
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
    double avg_width, avg_length, m_over_px_x, m_over_px_y, m_over_px_z, max_z, min_z = 0;
    while (points.size() > N) {
        if (show_process) {
            projectPointsTo2D(avg_width, avg_length, m_over_px_x, m_over_px_y, m_over_px_z, max_z, min_z);
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

void Space::projectPointsTo2D (double & avg_width, double & avg_length,
                               double & m_over_px_x, double & m_over_px_y, double & m_over_px_z,
                               double & max_z, double & min_z) {

    double height = 2000.;
    double width = 3000.;
    double max_radius = 50.;

    auto highest_energy_point = points[0];
    Eigen::Vector3d total{0, 0, 0};
    for (const auto &point: points) {
        if (point->total_energy > highest_energy_point->total_energy) {
            highest_energy_point = point;
        }
        total += point->position;
    }

    if (abs(avg_width) <= 0.0000000001 ) {
        Eigen::Vector3d average = total / points.size();
        avg_width = average[0];
        avg_length = average[2];

        double farthest_x = 0.;
        double farthest_y = 0.;
        max_z = -1000;
        min_z = 1000;
        for (const auto &point: points) {
            double x_dist = abs(avg_width - point->position[0]);
            double y_dist = abs(avg_length - point->position[2]);
            double h = point->position[1];
            if (x_dist > farthest_x) farthest_x = x_dist;
            if (y_dist > farthest_y) farthest_y = y_dist;
            if (h > max_z) {
                max_z = h;
            } else if (h < min_z) {
                min_z = h;
            }
        }

        m_over_px_x = (farthest_x / (width / 2. - max_radius));
        m_over_px_y = (farthest_y / (height / 2. - max_radius));
        m_over_px_z = (max_z - min_z) * 1.2 / max_radius;
    }

    double sep = max_z - min_z;

    cv::Mat canvasImage(int(height), int(width), CV_8UC3, cv::Scalar(255., 255., 255.));

    cv::Point2d im_center {width / 2, height / 2};
    for (const auto & point : points) {

        // get center point
        cv::Point2d c_pt {
                im_center.x + ((point->position[0] - avg_width) / m_over_px_x),
                im_center.y + ((point->position[2] - avg_length) / m_over_px_y)
        };

        // get radius
        double radius = (point->position[1] - min_z + sep*.2) / m_over_px_z;

        // get color
        cv::Scalar color;
        if (point != highest_energy_point) {
            color = cv::Scalar (0., 0., 0.);
        }

        // draw center and direction
        cv::circle(canvasImage, c_pt, int (radius), color, -1);
    }

    cv::Point2d q_pt {
            im_center.x + ((this->query[0] - avg_width) / m_over_px_x),
            im_center.y + ((this->query[2] - avg_length) / m_over_px_y)
    };

    double radius_q = (this->query[1] - min_z + sep*.2) / m_over_px_z;
    cv::circle(canvasImage, q_pt, int (radius_q), cv::Scalar (255., 0., 0.), -1);

    cv::Point2d c_pt {
            im_center.x + ((highest_energy_point->position[0] - avg_width) / m_over_px_x),
            im_center.y + ((highest_energy_point->position[2] - avg_length) / m_over_px_y)
    };

    double radius = (highest_energy_point->position[1] - min_z + sep*.2) / m_over_px_z;
    cv::circle(canvasImage, c_pt, int (radius), cv::Scalar (0., 255., 0.), -1);

    imshow("Spacing with " + to_string(points.size()) + " points remaining.", canvasImage);
    cv::waitKey(0);
}

