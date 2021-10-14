//
// Created by Cameron Fiore on 7/5/21.
//

#ifndef IMAGEMATCHERPROJECT_ORB_IMAGEMATCHER_H
#define IMAGEMATCHERPROJECT_ORB_IMAGEMATCHER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <utility>
#include <dirent.h>
#include <stdio.h>
#include <sys/time.h>

#include "DBoW2.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#if CV24
#include <opencv2/features2d/features2d.hpp>
#endif

using namespace std;
using namespace DBoW2;

class imageMatcher
{
public:
    //Default Constructor
    imageMatcher();

    //Constructor for static test. load query image from folder
    imageMatcher(string query_image_path, string indexed_image_list_file,
                 string database_file_path, string feature_name,
                 string matching_method);

    //Constructor for practicl use. query image as Mat file
    imageMatcher(string indexed_image_list_file, string database_file_path, string feature_name, string matching_method);

    ~imageMatcher();

    //Name Loader
    vector<string> list_names(const char* target_path);
    vector<string> load_image_list(string path);

    //Initial Matcher
    void changeStructure(const vector<float> &plain, vector<vector<float> > &out, int L);
    vector<pair<int, float> > matching_one_image(string query_image);
    vector<pair<int, float> > matching_one_image(cv::Mat image);

    vector<pair<int, int> > matching_images(cv::Mat image1, cv::Mat image2);
    vector<pair<int, int> > matching_images(string query1, string query2);

    vector<pair<int, float> > thresholding(vector<pair<int,float> > full_list, float threshold);

private:
    string query_Images_Path_;
    string feature_name_;
    string matching_method_;
    string indexed_image_information_;

    vector<string> indexed_image_list_;
    vector<string> query_list_;

    Surf64Database db_;
    cv::Ptr<cv::xfeatures2d::SURF> surf;

};


#endif //IMAGEMATCHERPROJECT_ORB_IMAGEMATCHER_H
