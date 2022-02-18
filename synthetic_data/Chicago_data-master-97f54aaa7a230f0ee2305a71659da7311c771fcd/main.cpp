#include <iostream>
#include <math.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <complex>
#include "dataGenerator.h"
#include <time.h>
#include <iomanip> 
#include <string.h>
#include <iomanip>
using namespace std;


static inline void dquat(const complex<double> a[4], const double b[4], double d[4])
 {
    
     *d++ =  a[0].real() * b[0] + a[1].real() * b[1] + a[2].real() * b[2] + a[2].real() * b[2];
     *d++ = -a[0].real() * b[1] + a[1].real() * b[0] - a[2].real() * b[3] + a[3].real() * b[2];
     *d++ = -a[0].real() * b[2] + a[2].real() * b[0] - a[3].real() * b[1] + a[1].real() * b[3];
     *d   = -a[0].real() * b[3] + a[3].real() * b[0] - a[1].real() * b[2] + a[2].real() * b[1];
  }


static inline double rotation_error(complex<double> p[4], double q[4])
{
	complex<double> norm_p = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
	double norm_q = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	for (int i = 0; i < 4; i++)
	{
		p[i] /= norm_p;
		q[i] /= norm_q;
	}

    double d[4];
    dquat(p, q , d);
	
    const double vnorm = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3]);
    return double(2) * std::atan2(vnorm, std::fabs(d[0]));
}


bool Evaluation(complex<double>* solutions, double* groundTruth)
{
	for (int i = 0; i < 312; i++)
	{
		int counter = i * 14;
		double error = rotation_error(solutions+counter, groundTruth);
		if (error < 0.000001);
		return 1;
	}
	return 0;
}



void chicagoDataBuilding(vector<singleView*> sampleViews, double* parameter, complex<double>* parameter_gammified, double* groundTruth)
{
	//NumParam = 56;
	//dimSolution = 14;
	generateParameters_chicago(sampleViews, parameter, groundTruth);		
	//gammify_chicago(parameter, parameter_gammified);
}

void prob_5pt2Views16UnknownsBuilding(vector<singleView*> sampleViews, double* parameter, double* groundTruth)
{
	generateParameters_5pt2Views16Unknowns(sampleViews,parameter,groundTruth);
}



void threeViewTriangulation(vector<singleView*> sampleViews, double* parameter, complex<double>* parameter_gammified, double* groundTruth, bool noise)
{
	//NumParam = 33;
	//dimSolution = 9;
	generateParameters_threeViewTriangulation(sampleViews, parameter, groundTruth, noise);
	gammify_threeViewTriangulation(parameter, parameter_gammified);
}


void nViewTriangulation(vector<singleView*> sampleViews, bool noise_t, bool noise_r, bool noise_point, int n, int ind)
{

	//Randomly Pick Points
	vector<int> perm(sampleViews[0]->getNum());
	for (int i = 0; i < sampleViews[0]->getNum(); i++)
	{
		perm[i] = i;
	}
	random_shuffle(perm.begin(), perm.end());

	//This will go over different points for 3 to N views 
	for (int nview = 3; nview <=n; nview++)
	{
		double *parameter = new double[(2*nview) + ((nview-1) * 9)];
		double *groundTruth = new double[2 * nview];


		
		//Generate parameter and ground truth
		generateParameters_NViewTriangulation(sampleViews, parameter, groundTruth, noise_point, noise_t, noise_r,nview, perm[0]);
	
		//Output parameters into files
		ofstream of("/home/hongyi/Documents/git_visionserver/Chicago_data/build/" + to_string(nview) + "/" + to_string(ind) + ".txt");
		if (of.is_open())
		{
			of << std::setprecision(20);
			for (int i = 0; i < (2 * nview) + ((nview-1) * 9); i++)
			{
				of << parameter[i] << " " <<  0.0f << endl;
			}
		}
		of.close();


		if (nview == n)
		{
			ofstream GT("/home/hongyi/Documents/git_visionserver/Chicago_data/build/GroundTruth/" + to_string(ind) + ".txt");
			if (GT.is_open())
			{
				for (int i = 0; i < nview * 2; i++)
				{
					GT << groundTruth[i] << endl;
				}
			}
			GT.close();
		}
		delete parameter;
		delete groundTruth;
	}

}

void threeAndFourViewTriangulation(vector<singleView*> sampleViews, double* parameter, complex<double>* parameter_gammified, double* groundTruth, bool noise, int ind)
{
	//Generate Parameters
	generateParameters_threeAndFourViewTriangulation(sampleViews, parameter, groundTruth, noise);
	gammify_threeAndFourViewTriangulation(parameter, parameter_gammified);	
	
	cout << "Finished" << endl;
	//Output Parameters and GrounTruth into files
	ofstream of3("/home/hongyi/Documents/git_visionserver/Chicago_data/build/threeView/" + to_string(ind) + ".txt");
	ofstream of4("/home/hongyi/Documents/git_visionserver/Chicago_data/build/fourView/" + to_string(ind) + ".txt");
	ofstream GT("/home/hongyi/Documents/git_visionserver/Chicago_data/build/GT/" + to_string(ind) + ".txt");

	if (of3.is_open())
	{
		of3 << std::setprecision(20);
		for (int i = 0; i < 33; i++)
		{
			of3 << parameter[i] << " " <<  0.0f << endl;
		}
	}
	of3.close();

	if (of4.is_open())
	{
		of4 << std::setprecision(20);
		for (int i = 0; i < 62; i++)
		{
			of4 << parameter[33 + i] << " " << 0.0f << endl;
		}
	}
	of4.close();

	if (GT.is_open())
	{
		for (int i = 0; i < 8; i++)
		{
			GT << groundTruth[i] << endl;
		}
}
	GT.close();
}


void readThreeView(string filename, complex<double>* parameter_gammified)
{
	ifstream if3(filename);
	int i = 0;

	cout << "Here" << endl;
	if (if3.is_open())
	{
		for (int j = 0; j < 33; j++)
		{
			double tempR, tempI; 
			if3 >> tempR;
			if3 >> tempI;
			parameter_gammified[i] = complex<double>(tempR, tempI);
			i++;
		}
	}
	if3.close();
}

void readFourView(string filename, complex<double>* parameter_gammified)
{
	ifstream if4(filename);
	int i = 0;
	if (if4.is_open())
	{
		for (int j = 0; j < 62; j++)
		{

			cout << i << endl;
			double tempR, tempI; 
			if4 >> tempR;
			if4 >> tempI;
			parameter_gammified[i] = complex<double>(tempR, tempI);
			i++;
		}
	}
	if4.close();
}



int main(int argc, char *argv[])
{
	//Generate Random Seed:
	srand (time(NULL));
	//Prepare Dataset
	string dataFolder = "/home/hongyi/Documents/git_visionserver/Chicago_data/synthcurves_dataset/spherical-ascii-100_views-perturb-radius_sigma10-normal_sigma0_01rad-minsep_15deg-no_two_cams_colinear_with_object/";
	ptsTgtDataset dataset = ptsTgtDataset(dataFolder);

	string problem;
	problem = std::string(argv[1]);
	cout << problem << endl;
	if (problem == "chicago")
	{
		cout << problem << endl;
		
		int numParam = 56;
		int dimSolution = 14;
		int nview = 3;
		vector<singleView*> sampleViews = dataset.get_items(nview);
		double *parameter = new double[numParam];
		double *groundTruth = new double[dimSolution];
		complex<double> *parameter_gammified = new complex<double>[numParam];
		chicagoDataBuilding(sampleViews, parameter, parameter_gammified, groundTruth);

		std::cout << "------------Paramaeters--------------" << std::endl;
		for (int i = 0; i < numParam; i++)
		{
			std::cout << parameter[i] << std::endl;
		}
		std::cout << "-------------Solutions--------Here is Wrong-----------" << std::endl;
		for (int i = 0; i < dimSolution; i++)
		{
			std::cout << groundTruth[i] << std::endl;
		}
	}
	else if (problem == "5pt2Views16Unknowns")
	{

		cout << problem << endl;
		int numParam = 20;
		int dimSolution = 6;
		int nview = 2;
		vector<singleView*> sampleViews = dataset.get_items(nview);
		double *parameter = new double[numParam];
		double *groundTruth = new double[dimSolution];
		prob_5pt2Views16UnknownsBuilding(sampleViews, parameter,  groundTruth);
		cout << std::setprecision(16);
		std::cout << "------------Paramaeters--------------" << std::endl;	
		for (int i = 0; i < numParam; i++)
		{
			std::cout << parameter[i] << std::endl;
		}
		std::cout << "-------------Solutions--------Here-----------" << std::endl;
		{
			for (int i = 0; i < 6; i++)
			{
				std::cout << groundTruth[i] << std::endl;
			}
		}

	}
	else
	{
		cout << "unknown" << endl; 

	}

//	int numExp = 1000;    //Number of Experiment
//	int numParam = 56;  // Number of Parameters
//	int dimSolution = 8;   // Dimension of Solution Space
	//ptsTgtDataset dataset = ptsTgtDataset(dataFolder);

	//Play with random number
//	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//	std::default_random_engine generator(seed);
//	std::normal_distribution<double> distribution(5.0,2.0);
//	for (int i = 0; i < 10; i++)	
//	{
//		double number = distribution(generator);
//		cout << number << endl;
//	}
//	complex<double> *parameter_gammified = new complex<double>[56];
//	double *parameter = new double[56];
//	double *groundTruth = new double[100];
//	for (int i = 0; i < 33; i++)
//	{
//		cout << parameter_gammified[i] << endl;
//	}
//	readFourView("/home/hongyi/Documents/git_visionserver/Chicago_data/build/fourView/0.txt",parameter_gammified);
//	for (int i = 0; i < 100; i++)
//		readFiles_targetParams("/home/hongyi/Documents/git_visionserver/Chicago_data/build/GT/" + to_string(i) + ".txt", parameter_gammified);

//
//	for (int i = 0; i < 62; i++)
//	{
//		cout << parameter_gammified[i] << endl;
//	}
////////////////////////////////////////////////////////////////////////////////////////////////
	//This is for genereating data for N-view triangulation problem
//	int nview = 3;
//	for (int i = 0; i < numExp; i++)
//	{
//		vector<singleView*> sampleViews = dataset.get_items(nview);
			
		//generateData_chicagoMinus(sampleViews);
//		chicagoDataBuilding(sampleViews, parameter, parameter_gammified, groundTruth);

//		std::cout << "------------Paramaeters--------------" << std::endl;
//		for (int i = 0; i < 56; i++)
//		{
//			std::cout << parameter[i] << std::endl;
//		}
//		std::cout << "------------------------------" << std::endl;
//		nViewTriangulation(sampleViews, true, true, true, nview, i);
//
//		for (int i = 0; i < sampleViews.size();i++)
//		{
//			delete sampleViews[i];
//		}
//	}

/////////////////////////////////////////////////////////////////////////////////////////////////

//	bool noise = 0;
//	for (int i = 0; i < numExp; i++)
//	{
//		double *parameter = new double[numParam];
//		complex<double> *parameter_gammified = new complex<double>[numParam];
//		double *groundTruth = new double[dimSolution];
//
//		for (int i = 0; i < dimSolution; i++)
//		{
//			groundTruth[i] = 0.0;
//		}
//		for (int i = 0; i < numParam; i++)
//		{
//			parameter[i] = 0.0;
//			parameter_gammified[i] = complex<double>(0.0,0.0);
//		}
//
//
//		complex<double> *result = new complex<double>[312 * dimSolution];
//		vector<singleView*> sampleViews = dataset.get_items(4);
//		threeAndFourViewTriangulation(sampleViews, parameter, parameter_gammified, groundTruth, noise, i);
//
//		
//		cout << "Value:" << endl;	
//		
//		cout.precision(23);
//		for (int i = 0; i < numParam; i++)
//		{
//			cout << parameter[i] << endl;
//		}
//
//		cout << "GT" << endl;
//		for (int i = 0; i < dimSolution; i++)
//		{
//			cout << groundTruth[i] << endl;
//		}
//
//
		//Put Solver Here:
//
//
//		//Evaluation
//		bool flag = Evaluation(result, groundTruth);	
//			
//		
//		
//		delete[] parameter;
//		delete[] groundTruth;
//		delete[] parameter_gammified;
//	

	
//	}
	return 0;
}
