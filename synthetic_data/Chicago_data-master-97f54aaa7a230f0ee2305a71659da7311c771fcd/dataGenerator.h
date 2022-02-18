#include <iostream>
#include <math.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include <complex>
#include <random>


using namespace std;

//Generating an array of complex random number
void randc(complex<double> *z)
{
	*z = complex<double>(rand()/double(RAND_MAX), rand()/double(RAND_MAX));
	*z /= sqrt((*z).real() * (*z).real() + (*z).imag() * (*z).imag());
};


//Gammifiy of Three view Triangulation
void gammify_threeViewTriangulation(double* parameter, complex<double>* parameter_gammified)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0,1.0);
	double number_real = distribution(generator);
	double number_imag = distribution(generator);


	for (int i = 0; i < 33; i++)
	{
		parameter_gammified[i] = parameter[i] * complex<double>(number_real,number_imag);
	}
}

//Gammifiy of Three view Triangulation
void gammify_threeAndFourViewTriangulation(double* parameter, complex<double>* parameter_gammified)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0,1.0);
	double number_real = distribution(generator);
	double number_imag = distribution(generator);


	for (int i = 0; i < 33+62; i++)
	{
		parameter_gammified[i] = parameter[i] * complex<double>(number_real,number_imag);
	}
}

//Copied and modified from Ricardo's Code
void gammify_chicago(double*  params, complex<double>* params_gammified)
{
  for (int i = 0; i < 56; i++)
  {
	  params_gammified[i] = complex<double>(params[i],0.0);
  }
  //  params = (diag0|diag1|diag2|diag3|diag4).*params;
  // diag0 --> pF in params ----------------------------------------------------
  complex<double> (*p)[3] = (complex<double> (*)[3]) params_gammified;
  complex<double> gammas[9]; 
  for (unsigned l=0; l < 9; ++l) {
    randc(gammas+l);
    const complex<double> &g = gammas[l];
    p[l][0] *= g; p[l][1] *= g; p[l][2] *= g;
  }
  
  // ids of two point-point lines at tangents
  static unsigned constexpr triple_intersect[6][2] = {{0,3},{0+1,3+1},{0+2,3+2},{0,6},{0+1,6+1},{0+2,6+2}};

  // diag1 --> pTriple in params -----------------------------------------------
  unsigned i = 9*3;
  for (unsigned tl=0; tl < 6; ++tl) {  // for each tangent line
    params_gammified[i++] *= std::conj(gammas[triple_intersect[tl][0]]);
    params_gammified[i++] *= std::conj(gammas[triple_intersect[tl][1]]);
  }
  
  // pChart gammas -------------------------------------------------------------
  complex<double> g;
  // diag2 -- tchart gamma
  randc(&g); for (unsigned k=0; k < 7; ++k) params_gammified[i++] *= g;
  // diag3 -- qchart, cam 2, gamma
  randc(&g); for (unsigned k=0; k < 5; ++k) params_gammified[i++] *= g;
  // diag4 -- qchart, cam 3, gamma
  randc(&g); for (unsigned k=0; k < 5; ++k) params_gammified[i++] *= g;
  //  p = (diag0|diag1|diag2|diag3|diag4).*p;
  //  total  27   12    7      5    5 = 56
}


 struct quat_shape { double w; double x; double y; double z; };
  static inline void rotm2quat(const double rr[9], double qq[4])
  {
    // use a struct to reinterpret q
    quat_shape *q = (quat_shape *) qq;
    const double (*r)[3] = (const double (*)[3]) rr;
    // coeff_eigen[i] = index in our quaternion shape of corresponding element
    static constexpr unsigned coeff_eigen[4] = {1, 2, 3, 0};

    // This algorithm comes from  "Quaternion Calculus and Fast Animation",
    // Ken Shoemake, 1987 SIGGRAPH course notes
    double t = rr[0] + rr[4] + rr[8]; // trace
    if (t > double(0)) {
      t = std::sqrt(t + double(1.0));
      q->w = double(0.5)*t;
      t = double(0.5)/t;
      q->x = (r[2][1] - r[1][2]) * t;
      q->y = (r[0][2] - r[2][0]) * t;
      q->z = (r[1][0] - r[0][1]) * t;
    } else {
      unsigned i = 0;
      if (r[1][1] > r[0][0]) i = 1;
      if (r[2][2] > r[i][i]) i = 2;
      unsigned j = (i+1)%3;
      unsigned k = (j+1)%3;
      t = std::sqrt(r[i][i]-r[j][j]-r[k][k] + double(1.0));
      qq[coeff_eigen[i]] = double(0.5) * t;
      t = double(0.5)/t;
      q->w = (r[k][j]-r[j][k])*t;
      qq[coeff_eigen[j]] = (r[j][i]+r[i][j])*t;
      qq[coeff_eigen[k]] = (r[k][i]+r[i][k])*t;
    }
//	cout << qq[0] << " " << qq[1] << " " << qq[2] << " " << qq[3] << endl;
  }

//Copied from Ricardo's code
//static inline void rotm2quat(const double rr[9], double qq[4])
//{
//    // use a struct to reinterpret q
//    double *q = qq;
//    const double (*r)[3] = (const double (*)[3]) rr;
//    // coeff_eigen[i] = index in our quaternion shape of corresponding element
//    static constexpr unsigned coeff_eigen[4] = {1, 2, 3, 0};
//
//    // This algorithm comes from  "Quaternion Calculus and Fast Animation",
//    // Ken Shoemake, 1987 SIGGRAPH course notes
//    double t = rr[0] + rr[4] + rr[8]; // trace
//    if (t > double(0)) {
//      t = std::sqrt(t + double(1.0));
//      q[0] = double(0.5)*t;
//      t = double(0.5)/t;
//      q[1] = (r[2][1] - r[1][2]) * t;
//      q[2] = (r[0][2] - r[2][0]) * t;
//      q[3] = (r[1][0] - r[0][1]) * t;
//    } else {
//      unsigned i = 0;
//      if (r[1][1] > r[0][0]) i = 1;
//      if (r[2][2] > r[i][i]) i = 2;
//      unsigned j = (i+1)%3;
//      unsigned k = (j+1)%3;
//      t = std::sqrt(r[i][i]-r[j][j]-r[k][k] + double(1.0));
//      qq[coeff_eigen[i]] = double(0.5) * t;
//      t = double(0.5)/t;
//      q[0] = (r[k][j]-r[j][k])*t;
//      qq[coeff_eigen[j]] = (r[j][i]+r[i][j])*t;
//      qq[coeff_eigen[k]] = (r[k][i]+r[i][k])*t;
//    }
//  }



//Tested
void rand_sphere(double *v, unsigned n)
{
	double m = 0;
	for (int i = 0; i < n; i++)
	{
		double r = ((rand() / double(RAND_MAX)) - 0.5) * 2;
		v[i] = r;
		m += r * r;
	}

	m = sqrt(m);
	for (int i = 0; i < n; i++)
		v[i] /= m;
}


//The class for single View from Synthetic Dataset
class singleView
{
public:
	singleView(){};
	singleView(string intrinsic, string singleViewCamera, string singleViewPts, string singleViewTgts)
	{
		readCamera(intrinsic, singleViewCamera);
		readPts(singleViewPts);
		readTgts(singleViewTgts);
	};
	
	~singleView()
	{
		delete[] _R;
		delete[] _K;
		delete[] _C;
		
		for (int i = 0; i < _Pts.size(); i++)
		{
			delete[] _Pts[i];
			delete[] _Tgts[i];
		}
	}
	
	void readCamera(string intrinsic, string singleViewCamera)
	{
		_K = new double[9];
		_R = new double[9];
		_C = new double[3];
		ifstream cameraFile(intrinsic);
		if (!cameraFile.is_open())
		{
			cout << "Wrong camera file" << endl;
		}
		else
		{
			for (int i = 0; i < 9; i++)
			{
				double singleValue;
				cameraFile >> singleValue;
				_K[i] = (singleValue);
			}
	
		}

		int counter = 0;
		

		cameraFile.close();
		
		ifstream RTFile(singleViewCamera);
		if (!RTFile.is_open()) 
		{
			cout << "Wrong RT file" << endl;
		}
		else
		{
			for (int i = 0; i < 9; i++)
			{
				RTFile >> _R[i];
			}
			for (int i = 0; i < 3; i++)
			{
				RTFile >> _C[i];
			}
		}
		RTFile.close();		
	};
	void readPts(string singleViewPts)
	{
		ifstream ptsFile(singleViewPts);
		if (!ptsFile.is_open())
		{
			cout << "Wrong Point File" << endl;
		}
		else
		{
			while(ptsFile.peek() != EOF)
			{
				double* pts_temp = new double[3];
				ptsFile >> pts_temp[0];
				ptsFile >> pts_temp[1];
				pts_temp[2] = 1.0;;
				_Pts.push_back(pts_temp);
			}
		}
		_numPoints = _Pts.size();
	};

	void readTgts(string singleViewTgts)
	{
		ifstream tgtsFile(singleViewTgts);
		if (!tgtsFile.is_open())
		{
			cout << "Wrong Point File" << endl;
		}
		else
		{
			while(tgtsFile.peek() != EOF)
			{
				double* tgts_temp = new double[3];
				tgtsFile >> tgts_temp[0];
				tgtsFile >> tgts_temp[1];
				tgts_temp[2] = 0.0;
				_Tgts.push_back(tgts_temp);
			}
		}
	};



	double* getR() {return _R;};
	double* getC() {return _C;};
	double* getK() {return _K;};
	int getNum() {return _numPoints;};
	double* getPoint(int n) { return _Pts[n];};
	double* getTangent(int n) {return _Tgts[n];};
	void printK()
	{ 
		int counter = 0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cout << _K[counter++] << " ";
			}
			cout << endl;
		}
	};
	void printR()
	{
		int counter = 0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cout << _R[counter++] << " ";
			}

			cout << endl;
		}

	};
	void printC()
	{
		for (int i = 0; i < 3; i++)
		{
			cout << _C[i] << endl;
		}

	};
	void printPoints(){};
	void printTangents(){};
private:
	int _numPoints;
	double* _K = NULL;
	double* _R = NULL;
	double* _C = NULL;
	vector<double*> _Pts;
	vector<double*> _Tgts;
};


//The data structure of Sythetic Curve Dataset 
class ptsTgtDataset
{
public:
	ptsTgtDataset(){};
	ptsTgtDataset(string data_folder)
	{
		//List files
		_numViews = 100;
		_intrinsicFile = data_folder + "calib.intrinsic";	
		for (int i = 0; i < 100; i++)
		{
			string numTemp = string(4 - to_string(i).size(), '0') + to_string(i);
			_cameraFiles.push_back(data_folder + "frame_" + numTemp + ".extrinsic");
			_ptsFiles.push_back(data_folder + "frame_" + numTemp + "-pts-2D.txt");
			_tgtsFiles.push_back(data_folder + "frame_" + numTemp + "-tgts-2D.txt");
		}
	};

	vector<singleView*> get_items(int n)
	{
		vector<singleView*> return_vector;
		vector<int> perm(_numViews);
		for (int i = 0; i < _numViews; i++)
		{
			perm[i] = i;
		}
		random_shuffle(perm.begin(), perm.end());	
		
		//Test Code
		perm[0] = 42;
		perm[1] = 54;
		perm[2] = 62;

		for (int i = 0; i < n; i++)
		{
			//Pick Random Views
			//cout << perm[i] << endl;
			singleView* oneView = new singleView(_intrinsicFile, _cameraFiles[perm[i]], _ptsFiles[perm[i]], _tgtsFiles[perm[i]]);
			return_vector.push_back(oneView);	
		}
		return return_vector;
	};
private:
	int _numViews = 100;
	string _intrinsicFile;
	vector<string> _ptsFiles;
	vector<string> _tgtsFiles;
	vector<string> _cameraFiles;
};

//Getting the homogeneous representation of a line constructed with p1 and p2;
void generateLineFromTwoPoints(double* p1, double* p2, double* out)
{

	cout << "****************************************" << endl;
	cout << p1[0] << " " << p1[1] << " " << p1[2] << endl;
	cout << p2[0] << " " << p2[1] << " " << p2[2] << endl;

	out[0] = p1[1] * p2[2] - p1[2] * p2[1];
	out[1] = -(p1[0] * p2[2] - p1[2] * p2[0]);
	out[2] = p1[0] * p2[1] - p1[1] * p2[0];
	cout << out[0] << " " << out[1] << " " << out[2] << endl;

	double A = sqrt(out[0] * out[0] + out[1] * out[1]);
	cout << A << endl;	

	out[0] = out[0] / A;
	out[1] = out[1] / A;
	out[2] = out[2] / A;
	cout << out[0] << " " << out[1] << " " << out[2] << endl;
	cout << "***********************************************" << endl;
}

//For chicago method, build 5 lines with three points and three tangents
void getLines(double* p1, double* p2, double* p3, double* t1, double* t2, double* t3, double* K, double** return_vector)
{
	double pA[3], pB[3], pC[3];
	
	
	pA[0] = (1/K[0]) * p1[0] - (K[2] / K[0]);
	pA[1] = (1/K[4]) * p1[1] - (K[5] / K[4]);
	pA[2] = 1.0;
	
	pB[0] = (1/K[0]) * p2[0] - (K[2] / K[0]);
	pB[1] = (1/K[4]) * p2[1] - (K[5] / K[4]);
	pB[2] = 1.0;

	pC[0] = (1/K[0]) * p3[0] - (K[2] / K[0]);
	pC[1] = (1/K[4]) * p3[1] - (K[5] / K[4]);
	pC[2] = 1.0;


	double l1[3], l2[3], l3[3];
	generateLineFromTwoPoints(pA,pB,l1);
	generateLineFromTwoPoints(pA,pC,l2);
	generateLineFromTwoPoints(pB,pC,l3);
	
//	cout << "p1:" << p1[0] << " " << p1[1] << " " << p1[2] << endl;
//	cout << "pA:" << pA[0] << " " << pA[1] << " " << pA[2] << endl;
//	cout << "p2:" << p2[0] << " " << p2[1] << " " << p2[2] << endl;
//	cout << "pB:" << pB[0] << " " << pB[1] << " " << pB[2] << endl;
//
//	cout << "l1:" << l1[0] << " " << l1[1] << " " << l1[2] << endl;
	double newPoint[3], newPoint2[3];
	for (int i = 0; i < 3; i++)
	{
		newPoint[i] = pA[i] + t1[i];
		newPoint2[i] = pB[i] + t2[i];
	}
	double l4[3], l5[3];
	generateLineFromTwoPoints(pA,newPoint,l4);
	generateLineFromTwoPoints(pB,newPoint2,l5);
	

	return_vector[0][0] = l1[0];
	return_vector[1][0] = l1[1];
	return_vector[2][0] = l1[2];
	return_vector[0][1] = l2[0];
	return_vector[1][1] = l2[1];
	return_vector[2][1] = l2[2];
	return_vector[0][2] = l3[0];
	return_vector[1][2] = l3[1];
	return_vector[2][2] = l3[2];
	return_vector[0][3] = l4[0];
	return_vector[1][3] = l4[1];
	return_vector[2][3] = l4[2];
	return_vector[0][4] = l5[0];
	return_vector[1][4] = l5[1];
	return_vector[2][4] = l5[2];
}

//Build parameters from the lines especially for the Chicago problem
void lines2Params(double** pLines, double* parameters)
{
	//pDouble part
	int counter = 0;
	for (int i = 0; i < 9; i++)
		for (int j = 0; j < 3; j++)
			parameters[counter++] = pLines[i][j];
	static unsigned constexpr triple_intersections[6][3] =
    {{0,3,9},{0+1,3+1,9+1},{0+2,3+2,9+2},{0,6,12},{0+1,6+1,12+1},{0+2,6+2,12+2}};

	counter = 0;
	for (int i = 0; i < 6; i++)
	{
		double pStack[3][3];
		//Matrix<double,3,3> pStack;
		pStack[0][0] = pLines[triple_intersections[i][0]][0];
		pStack[0][1] = pLines[triple_intersections[i][0]][1];
		pStack[0][2] = pLines[triple_intersections[i][0]][2];

		pStack[1][0] = pLines[triple_intersections[i][1]][0];
		pStack[1][1] = pLines[triple_intersections[i][1]][1];
		pStack[1][2] = pLines[triple_intersections[i][1]][2];

		pStack[2][0] = pLines[triple_intersections[i][2]][0];
		pStack[2][1] = pLines[triple_intersections[i][2]][1];
		pStack[2][2] = pLines[triple_intersections[i][2]][2];


//		pStack.row(0) = pLines.row(triple_intersections[i][0]);
//		pStack.row(1) = pLines.row(triple_intersections[i][1]);
//		pStack.row(2) = pLines.row(triple_intersections[i][2]);
		double l0l0 = pStack[0][0] * pStack[0][0] + pStack[0][1] * pStack[0][1] + pStack[0][2] * pStack[0][2];
		double l0l1 = pStack[0][0] * pStack[1][0] + pStack[0][1] * pStack[1][1] + pStack[0][2] * pStack[1][2];
		double l1l1 = pStack[1][0] * pStack[1][0] + pStack[1][1] * pStack[1][1] + pStack[1][2] * pStack[1][2];
		double l2l0 = pStack[2][0] * pStack[0][0] + pStack[2][1] * pStack[0][1] + pStack[2][2] * pStack[0][2];
		double l2l1 = pStack[2][0] * pStack[1][0] + pStack[2][1] * pStack[1][1] + pStack[2][2] * pStack[1][2];



//		double l0l0 = pStack.row(0).dot(pStack.row(0));
//		double l0l1 = pStack.row(0).dot(pStack.row(1));
//		double l1l1 = pStack.row(1).dot(pStack.row(1));
//		double l2l0 = pStack.row(2).dot(pStack.row(0));
//		double l2l1 = pStack.row(2).dot(pStack.row(1));

		double v1[3]; v1[0] = l0l0; v1[1] = l0l1; v1[2] = l2l0;
		double v2[3]; v2[0] = l0l1; v2[1] = l1l1; v2[2] = l2l1;
		double l2_l0l1[3];

		l2_l0l1[0] = v1[1] * v2[2] - v1[2] * v2[1];
		l2_l0l1[1] = -(v1[0] * v2[2] - v1[2] * v2[0]);
		l2_l0l1[2] = v1[0] * v2[1] - v1[1] * v2[0];

//		v1.cross(v2);

		parameters[27+(counter++)] = l2_l0l1[0] / l2_l0l1[2];
		parameters[27+(counter++)] = l2_l0l1[1] / l2_l0l1[2]; 
	
	}	

	rand_sphere(parameters+27+12,7);
	rand_sphere(parameters+27+12+7,5);
	rand_sphere(parameters+27+12+7+5,5);

}

//Rotation matrix multiplication
void RRMulti(double inA[][3], double inB[][3], double out[][3], int rowA, int colA, int rowB, int colB)
{
	for(int i=0; i<rowA; ++i)
		for(int j=0; j<colB; ++j)
			for(int k=0; k<colA; ++k)
    			out[i][j]+=inA[i][k]*inB[k][j];
};

//Multiplication between R(3*3) and T(3*1)
void RTMulti(double inA[][3], double inB[3], double out[3], int rowA, int colA, int rowB, int colB)
{
	for(int i=0; i<rowA; ++i)
		for(int j=0; j<colB; ++j)
			for(int k=0; k<colA; ++k)
    			out[i]+=inA[i][k]*inB[k];
};

//Building a cross product matrix, a.k.a. T to [T]_x
void crossProductMat(double in[3], double out[][3])
{
	out[0][0] = 0.0, out[1][1] = 0.0, out[2][2] = 0.0;
	out[0][1] = -in[2], out[0][2] = in[1], out[1][2] = -in[0];
	out[1][0] = in[2], out[2][0] = -in[1], out[2][1] = in[0];
}

//Transpose of a matrix
void matTranspose(double in[][3], double out[][3], int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			out[i][j] = in[j][i];
		}
	}
};

//Subtraction of 3-dim vectors
void vecSubs(double inA[3], double inB[3], double out[3], int row, int col)
{
	for (int i = 0; i < row; i++)
			out[i] = inA[i] - inB[i];
}


//Addition of 3-dim vectors
void vecAdd(double inA[3], double inB[3], double out[3], int row, int col)
{
	for (int i = 0; i < row; i++)
			out[i] = inA[i] + inB[i];
}

void EularNoiseToRotationMat(double x, double y, double z, double out[][3])
{
	double Rx[3][3];
	double Ry[3][3];
	double Rz[3][3];
	double output_temp[3][3];
	double output_temp1[3][3];
	double output_temp2[3][3];

	for (int i = 0; i < 9; i++)
	{
		Rx[i / 3][i % 3] = 0.0f;
		Ry[i / 3][i % 3] = 0.0f;
		Rz[i / 3][i % 3] = 0.0f;
		output_temp[i / 3][i % 3] = 0.0f;
		output_temp1[i / 3][i % 3] = 0.0f;
		output_temp2[i / 3][i % 3] = 0.0f;

	}

	Rx[0][0] = 1.0f; Rx[1][1] = cos(x); Rx[1][2] = -sin(x); Rx[2][1] = sin(x); Rx[2][2] = cos(x);
	Ry[1][1] = 1.0f; Ry[0][0] = cos(y); Ry[0][2] = sin(y); Ry[2][0] = -sin(y); Ry[2][2] = cos(y);
	Rz[2][2] = 1.0f; Rz[0][0] = cos(z); Rz[0][1] = -sin(z); Rz[1][0] = sin(z); Rz[1][1] = cos(z);

//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << out[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;

//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << Rx[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;
//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << Ry[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;
//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << Rz[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;


	RRMulti(Rz,out,output_temp, 3,3,3,3);
//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << output_temp[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;

	RRMulti(Ry,output_temp,output_temp1,3,3,3,3);
//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << output_temp1[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;

	RRMulti(Rx,output_temp1,output_temp2,3,3,3,3);
//	for (int temp = 0; temp < 9; temp++)
//	{
//		cout << output_temp2[temp / 3][temp % 3] << " ";
//	}
//	cout << endl;
	
	for (int temp = 0; temp < 9; temp++)
	{
		out[temp / 3][temp % 3] = output_temp2[temp / 3][temp % 3];
	}
}

void generateParameters_NViewTriangulation(vector<singleView*> views, double* parameter, double* groundTruth, bool noise, bool noise_t, bool noise_R ,int n, int perm)
{	
	
	//Create Random Seed for adding point noise	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0,1.0);

	//Build Points for each view
	int counter = 0;
	for (int i = 0; i < n; i++)
	{
		double noise_x = 0;
		double noise_y = 0;

		if (noise == 1)
		{
			noise_x = distribution(generator);
			noise_y = distribution(generator);
		}
		
		double *p_1;
		double pA[3];
		p_1 = views[i]->getPoint(perm);
		pA[0] = (1/(views[0]->getK()[0])) * (p_1[0] + noise_x) - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pA[1] = (1/(views[0]->getK()[4])) * (p_1[1] + noise_y) - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pA[2] = 1.0;


		groundTruth[counter] = (1/(views[0]->getK()[0])) * (p_1[0]) - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		groundTruth[counter + 1] = (1/(views[0]->getK()[4])) * (p_1[1]) - ((views[0]->getK()[5]) / (views[0]->getK()[4]));

		//Put the points into parameter vector,
		//The structure of the parameter vector is:
		// 9 * (n-1) for essential matrices / fundamental matrices
		// 2 * n for image points
		parameter[9*(n-1) + counter] = pA[0];
		parameter[9*(n-1) + counter + 1] = pA[1];
		counter += 2;
	}
	

	//Build Essential Matrix between each views
	int view1 = 0;
	int view2 = 1;
	counter = 0;
	for (int iview = 0; iview < n-1; iview++)
	{
		double (*R1)[3] = (double (*)[3])&views[view1]->getR()[0];
		double (*R2)[3] = (double (*)[3])&views[view2]->getR()[0];
	
		double R12[3][3];
		double T12[3];
		double C1_C2[3];
		
		for (int i = 0; i < 9; i++) 
		{
			R12[i/3][i%3] = 0.0;
		}
		//Initialization
		for (int i = 0; i < 3; i++) 
		{
			T12[i] = 0.0;	
		}
		double R1_inv[3][3], R2_inv[3][3];
		matTranspose(R1,R1_inv,3,3);
		matTranspose(R2,R2_inv,3,3);

		RRMulti(R2,R1_inv, R12,3,3,3,3);
	
		vecSubs(views[view1]->getC(),views[view2]->getC(),C1_C2,3,1);
		RTMulti(R2,C1_C2,T12, 3, 3, 3, 1);

		//cout << T12[0] * T12[0] + T12[1] * T12[1] + T12[2] * T12[2] << endl;	
		//Perturb Ts if needed
		if (noise_t == true)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::normal_distribution<double> distribution(0.0,4.0);
			for (int i = 0; i < 3; i++)
			{
				//cout << distribution(generator) << endl;
				T12[i] += distribution(generator);
			}
		}
		
		if (noise_R == true)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::normal_distribution<double> distribution(0.0,0.02);

//			for (int temp = 0; temp < 9; temp++)
//			{
//				cout << R12[temp / 3][temp % 3] << " ";
//			}
//			cout << endl;

			EularNoiseToRotationMat(distribution(generator), distribution(generator), distribution(generator), R12);
			
//			for (int temp = 0; temp < 9; temp++)
//			{
//				cout << R12[temp / 3][temp % 3] << " ";
//			}
//			cout << endl;

		}

		//Compute Essential Matrices:
		double Tx12[3][3];
		double E12[3][3];
	
		for (int i = 0; i < 9; i++) {
			Tx12[i/3][i%3] = 0.0;
			E12[i/3][i%3] = 0.0;
		}

		crossProductMat(T12,Tx12);
		
		RRMulti(Tx12,R12,E12,3,3,3,3);
	
		//Put E into parameters
		for (int i = 0; i < 9;i++)
		{
			parameter[counter + i] = E12[i % 3][i / 3];
		}
		counter += 9;
		view1++;
		view2++;
	}
}

void generateParameters_threeAndFourViewTriangulation(vector<singleView*> views, double* parameter, double* groundTruth, bool noise)
{
	//Generate E12, E23, E13, E14, E24, E34 
	//First generate R12, R23, R13, R14, R24, R34 
	//Then T12, T23, T13, T14, T24, T34
	double (*R1)[3] = (double (*)[3])&views[0]->getR()[0];
	double (*R2)[3] = (double (*)[3])&views[1]->getR()[0];
	double (*R3)[3] = (double (*)[3])&views[2]->getR()[0];
	double (*R4)[3] = (double (*)[3])&views[3]->getR()[0];


	double R12[3][3], R23[3][3], R13[3][3], R14[3][3], R24[3][3], R34[3][3];
	double T12[3], T23[3], T13[3], T14[3], T24[3], T34[3];
	double C1_C2[3], C2_C3[3], C1_C3[3], C1_C4[3], C2_C4[3], C3_C4[3];
	
	for (int i = 0; i < 9; i++) {
		R12[i/3][i%3] = 0.0;
		R23[i/3][i%3] = 0.0;
		R13[i/3][i%3] = 0.0;
		R14[i/3][i%3] = 0.0;
		R24[i/3][i%3] = 0.0;
		R34[i/3][i%3] = 0.0;
	}

	//Initialization
	for (int i = 0; i < 3; i++) {
		T12[i] = 0.0;
		T23[i] = 0.0;
		T13[i] = 0.0;
		T14[i] = 0.0;
		T24[i] = 0.0;
		T34[i] = 0.0;
		C1_C2[i] = 0.0;
		C2_C3[i] = 0.0;
		C1_C3[i] = 0.0;
		C1_C4[i] = 0.0;
		C2_C4[i] = 0.0;
		C3_C4[i] = 0.0;
	}
	
	double R1_inv[3][3], R2_inv[3][3], R3_inv[3][3], R4_inv[3][3];
	matTranspose(R1,R1_inv,3,3);
	matTranspose(R2,R2_inv,3,3);
	matTranspose(R3,R3_inv,3,3);
	matTranspose(R4,R4_inv,3,3);

	RRMulti(R2,R1_inv, R12,3,3,3,3);
	RRMulti(R3,R2_inv, R23,3,3,3,3);
	RRMulti(R3,R1_inv, R13,3,3,3,3);
	RRMulti(R4,R1_inv, R14,3,3,3,3);
	RRMulti(R4,R2_inv, R24,3,3,3,3);
	RRMulti(R4,R3_inv, R34,3,3,3,3);
	
	
	vecSubs(views[0]->getC(),views[1]->getC(),C1_C2,3,1);
	RTMulti(R2,C1_C2,T12, 3, 3, 3, 1);
	vecSubs(views[1]->getC(),views[2]->getC(),C2_C3,3,1);
	RTMulti(R3,C2_C3,T23,3,3,3,1);
	vecSubs(views[0]->getC(),views[2]->getC(),C1_C3,3,1);
	RTMulti(R3,C1_C3,T13,3,3,3,1);
	vecSubs(views[0]->getC(),views[3]->getC(),C1_C4,3,1);
	RTMulti(R4,C1_C4,T14,3,3,3,1);
	vecSubs(views[1]->getC(),views[3]->getC(),C2_C4,3,1);
	RTMulti(R4,C2_C4,T24,3,3,3,1);
	vecSubs(views[2]->getC(),views[3]->getC(),C3_C4,3,1);
	RTMulti(R4,C3_C4,T34,3,3,3,1);

	cout << "T13 New way:" <<T13[0] << " " << T13[1] << " " << T13[2] << endl;

	double T_temp[3];
	RTMulti(R23,T12,T_temp,3,3,3,1);
	vecAdd(T_temp,T23,T13,3,1);

	cout << "T13 Old way:" <<T13[0] << " " << T13[1] << " " << T13[2] << endl;
	
	//Perturb Ts
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0,2.0);
	for (int i = 0; i < 3; i++)
	{
		T12[i] += distribution(generator);
		T23[i] += distribution(generator);
		T13[i] += distribution(generator);
		T14[i] += distribution(generator);
		T24[i] += distribution(generator);
		T34[i] += distribution(generator);
	}

	//Compute 6 Essential Matrices:
	double Tx12[3][3], Tx13[3][3], Tx23[3][3], Tx14[3][3], Tx24[3][3], Tx34[3][3];
	double E12[3][3], E23[3][3], E13[3][3], E14[3][3], E24[3][3], E34[3][3];
	
	for (int i = 0; i < 9; i++) {
		Tx12[i/3][i%3] = 0.0;
		Tx23[i/3][i%3] = 0.0;
		Tx13[i/3][i%3] = 0.0;
		Tx14[i/3][i%3] = 0.0;
		Tx24[i/3][i%3] = 0.0;
		Tx34[i/3][i%3] = 0.0;
		E12[i/3][i%3] = 0.0;
		E23[i/3][i%3] = 0.0;
		E13[i/3][i%3] = 0.0;
		E14[i/3][i%3] = 0.0;
		E24[i/3][i%3] = 0.0;
		E34[i/3][i%3] = 0.0;
	}

	crossProductMat(T12,Tx12);
	crossProductMat(T23,Tx23);
	crossProductMat(T13,Tx13);
	crossProductMat(T14,Tx14);
	crossProductMat(T24,Tx24);
	crossProductMat(T34,Tx34);

	RRMulti(Tx12,R12,E12,3,3,3,3);
	RRMulti(Tx13,R13,E13,3,3,3,3);
	RRMulti(Tx23,R23,E23,3,3,3,3);
	RRMulti(Tx14,R14,E14,3,3,3,3);
	RRMulti(Tx24,R24,E24,3,3,3,3);
	RRMulti(Tx34,R34,E34,3,3,3,3);

	//Output Parameter for Three View Triangulation problem
	//Now out of 33, 27 are here, Insert them into parameter array
	for (int i = 0; i < 9; i++)
	{
		parameter[i] = E12[i % 3][i / 3];
		parameter[i+9] = E23[i % 3][i / 3];
		parameter[i+9+9] = E13[i % 3][i / 3];
	}


	//Extract a random point from three views
	//Permutation of point id
	vector<int> perm(views[0]->getNum());
	for (int i = 0; i < views[0]->getNum(); i++)
	{
		perm[i] = i;
	}
	random_shuffle(perm.begin(), perm.end());
	
	cout << perm[0] << endl;	
	double *p_1, *p_2, *p_3, *p_4;
	double pA[3], pB[3], pC[3], pD[3];
	p_1 = views[0]->getPoint(perm[0]);	
	p_2 = views[1]->getPoint(perm[0]);
	p_3 = views[2]->getPoint(perm[0]);
	p_4 = views[3]->getPoint(perm[0]);
	//Transfer points from pixel to meters
	pA[0] = (1/(views[0]->getK()[0])) * p_1[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pA[1] = (1/(views[0]->getK()[4])) * p_1[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pA[2] = 1.0;
	pB[0] = (1/(views[0]->getK()[0])) * p_2[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pB[1] = (1/(views[0]->getK()[4])) * p_2[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pB[2] = 1.0;
	pC[0] = (1/(views[0]->getK()[0])) * p_3[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pC[1] = (1/(views[0]->getK()[4])) * p_3[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pC[2] = 1.0;
	pD[0] = (1/(views[0]->getK()[0])) * p_4[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pD[1] = (1/(views[0]->getK()[4])) * p_4[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pD[2] = 1.0;

	//Put points into ground truth 
	for (int i = 0; i < 2; i++)
	{
		groundTruth[i] = pA[i];
		groundTruth[i+2] = pB[i];
		groundTruth[i+2+2] = pC[i];
		groundTruth[i+2+2+2] = pD[i];
	}


	//Add Gaussian Noise to image points
	if (noise == 1)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(0.0,1.0);

		for (int i = 0; i < 2; i++)
		{
			double number = distribution(generator);
			p_1[i] = p_1[i] + number;
			number = distribution(generator);
			p_2[i] = p_2[i] + number;
			number = distribution(generator);
			p_3[i] = p_3[i] + number;
			number = distribution(generator);
			p_4[i] = p_4[i] + number;
		}
		
		//Transfer points from pixel to meters
		pA[0] = (1/(views[0]->getK()[0])) * p_1[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pA[1] = (1/(views[0]->getK()[4])) * p_1[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pA[2] = 1.0;
		pB[0] = (1/(views[0]->getK()[0])) * p_2[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pB[1] = (1/(views[0]->getK()[4])) * p_2[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pB[2] = 1.0;
		pC[0] = (1/(views[0]->getK()[0])) * p_3[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pC[1] = (1/(views[0]->getK()[4])) * p_3[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pC[2] = 1.0;
		pD[0] = (1/(views[0]->getK()[0])) * p_4[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pD[1] = (1/(views[0]->getK()[4])) * p_4[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pD[2] = 1.0;

	}
	//Put points into parameter array
	for (int i = 0; i < 2; i++)
	{
		parameter[27+i] = pA[i]; 
		parameter[27+2+i] = pB[i];
		parameter[27+2+2+i] = pC[i];
	}

	//The 4 view triangulation parameters
	for (int i = 0; i < 9; i++)
	{
		parameter[i+33] = E12[i % 3][i / 3];
		parameter[i+33+9] = E23[i % 3][i / 3];
		parameter[i+33+9+9] = E13[i % 3][i / 3];
		parameter[i+33+9+9+9] = E14[i % 3][i/3];
		parameter[i+33+9+9+9+9] = E24[i % 3][i/3];
		parameter[i+33+9+9+9+9+9] = E34[i % 3][i/3];
	}
	for (int i = 0; i < 2;i++)
	{
		parameter[54 + 33 + i] = pA[i];
		parameter[54 + 33 + 2 + i] = pB[i];
		parameter[54 + 33 + 2 + 2 + i] = pC[i];
		parameter[54 + 33 + 2 + 2 + 2 + i] = pD[i];
	}

}


void generateParameters_threeViewTriangulation(vector<singleView*> views, double* parameter, double* groundTruth, bool noise)
{
	//Generate R12, R23, R13, T12, T23, T13;
	double (*R1)[3] = (double (*)[3])&views[0]->getR()[0];
	double (*R2)[3] = (double (*)[3])&views[1]->getR()[0];
	double (*R3)[3] = (double (*)[3])&views[2]->getR()[0];
	
	double R12[3][3], R23[3][3], R13[3][3];
	double T12[3], T23[3], T13[3];
	double C1_C2[3];
	double C2_C3[3];
	
	for (int i = 0; i < 9; i++) {
		R12[i/3][i%3] = 0.0;
		R23[i/3][i%3] = 0.0;
		R13[i/3][i%3] = 0.0;
	}

	for (int i = 0; i < 3; i++) {
		T12[i] = 0.0;
		T23[i] = 0.0;
		T13[i] = 0.0;
		C1_C2[i] = 0.0;
		C2_C3[i] = 0.0;
	}
	double R1_inv[3][3], R2_inv[3][3], R3_inv[3][3];
	matTranspose(R1,R1_inv,3,3);
	matTranspose(R2,R2_inv,3,3);
	matTranspose(R3,R3_inv,3,3);

	RRMulti(R2,R1_inv, R12, 3,3,3,3);
	RRMulti(R3,R2_inv, R23, 3,3,3,3);
	RRMulti(R23,R12,R13,3,3,3,3);
	
	vecSubs(views[0]->getC(),views[1]->getC(),C1_C2,3,1);
	RTMulti(R2,C1_C2,T12, 3, 3, 3, 1);
	vecSubs(views[1]->getC(),views[2]->getC(),C2_C3,3,1);
	RTMulti(R3,C2_C3,T23,3,3,3,1);	
		
	double T_temp[3];
	RTMulti(R23,T12,T_temp,3,3,3,1);
	vecAdd(T_temp,T23,T13,3,1);


	//Compute Three Essential Matrices E12 E23 E13
	double Tx12[3][3], Tx13[3][3], Tx23[3][3];
	double E12[3][3], E23[3][3], E13[3][3];
	
	for (int i = 0; i < 9; i++) {
		Tx12[i/3][i%3] = 0.0;
		Tx23[i/3][i%3] = 0.0;
		Tx13[i/3][i%3] = 0.0;
		E12[i/3][i%3] = 0.0;
		E23[i/3][i%3] = 0.0;
		E13[i/3][i%3] = 0.0;
	}

	crossProductMat(T12,Tx12);
	crossProductMat(T23,Tx23);
	crossProductMat(T13,Tx13);

	RRMulti(Tx12,R12,E12,3,3,3,3);
	RRMulti(Tx13,R13,E13,3,3,3,3);
	RRMulti(Tx23,R23,E23,3,3,3,3);
	//Now out of 33, 27 are here, Insert them into parameter array
	for (int i = 0; i < 9; i++)
	{
		parameter[i] = E12[i % 3][i / 3];
		parameter[i+9] = E23[i % 3][i / 3];
		parameter[i+9+9] = E13[i % 3][i / 3];
	}

	for (int i = 0; i < 9; i++)
	{
		cout << E12[i % 3][i / 3] << endl;;
	}
	//Extract a random point from three views
	//Permutation of point id
	vector<int> perm(views[0]->getNum());
	for (int i = 0; i < views[0]->getNum(); i++)
	{
		perm[i] = i;
	}
	random_shuffle(perm.begin(), perm.end());
	
	cout << perm[0] << endl;	
	double *p_1, *p_2, *p_3;
	double pA[3], pB[3], pC[3];
	p_1 = views[0]->getPoint(perm[0]);	
	p_2 = views[1]->getPoint(perm[0]);
	p_3 = views[2]->getPoint(perm[0]);

	//Transfer points from pixel to meters
	pA[0] = (1/(views[0]->getK()[0])) * p_1[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pA[1] = (1/(views[0]->getK()[4])) * p_1[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pA[2] = 1.0;
	pB[0] = (1/(views[0]->getK()[0])) * p_2[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pB[1] = (1/(views[0]->getK()[4])) * p_2[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pB[2] = 1.0;
	pC[0] = (1/(views[0]->getK()[0])) * p_3[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
	pC[1] = (1/(views[0]->getK()[4])) * p_3[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
	pC[2] = 1.0;

	//Put points into ground truth 
	for (int i = 0; i < 2; i++)
	{
		groundTruth[i] = pA[i];
		groundTruth[i+2] = pB[i];
		groundTruth[i+2+2] = pC[i];
	}
	groundTruth[6] = 1.0;
	groundTruth[7] = 1.0;
	groundTruth[8] = 1.0;
	
	//Add Gaussian Noise to image points
	if (noise == 1)
	{
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution(0.0,3.0);

		for (int i = 0; i < 2; i++)
		{
			double number = distribution(generator);
			p_1[i] = p_1[i] + number;
			number = distribution(generator);
			p_2[i] = p_2[i] + number;
			number = distribution(generator);
			p_3[i] = p_3[i] + number;
		}
		
		//Transfer points from pixel to meters
		pA[0] = (1/(views[0]->getK()[0])) * p_1[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pA[1] = (1/(views[0]->getK()[4])) * p_1[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pA[2] = 1.0;
		pB[0] = (1/(views[0]->getK()[0])) * p_2[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pB[1] = (1/(views[0]->getK()[4])) * p_2[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pB[2] = 1.0;
		pC[0] = (1/(views[0]->getK()[0])) * p_3[0] - ((views[0]->getK()[2]) / (views[0]->getK()[0]));
		pC[1] = (1/(views[0]->getK()[4])) * p_3[1] - ((views[0]->getK()[5]) / (views[0]->getK()[4]));
		pC[2] = 1.0;
	}
	//Put points into parameter array
	for (int i = 0; i < 2; i++)
	{
		parameter[27+i] = pA[i]; 
		parameter[27+2+i] = pB[i];
		parameter[27+2+2+i] = pC[i];
	}
}


void generateData_chicagoMinus(vector<singleView*> views)
{
	double (*R1)[3] = (double (*)[3])&views[0]->getR()[0];
	double (*R2)[3] = (double (*)[3])&views[1]->getR()[0];
	double (*R3)[3] = (double (*)[3])&views[2]->getR()[0];

	double (*C1)[1] = (double (*)[1])&views[0]->getC()[0];
	double (*C2)[1] = (double (*)[1])&views[1]->getC()[0];
	double (*C3)[1] = (double (*)[1])&views[2]->getC()[0];

	
	vector<int> perm(views[0]->getNum());
	for (int i = 0; i < views[0]->getNum(); i++)
	{
		perm[i] = i;
	}
	random_shuffle(perm.begin(), perm.end());


	//Test Code
	perm[0] = 0;
	perm[1] = 49;
	perm[2] = 149;

	//Output Points and Tangents
	double *p1_1, *p2_1, *p3_1, *p1_2, *p2_2, *p3_2, *p1_3, *p2_3, *p3_3;
	double *t1_1, *t2_1, *t3_1, *t1_2, *t2_2, *t3_2, *t1_3, *t2_3, *t3_3;
	p1_1 = views[0]->getPoint(perm[0]);
	p2_1 = views[0]->getPoint(perm[1]);
	p3_1 = views[0]->getPoint(perm[2]);
	
	p1_2 = views[1]->getPoint(perm[0]);
	p2_2 = views[1]->getPoint(perm[1]);
	p3_2 = views[1]->getPoint(perm[2]);

	p1_3 = views[2]->getPoint(perm[0]);
	p2_3 = views[2]->getPoint(perm[1]);
	p3_3 = views[2]->getPoint(perm[2]);

	t1_1 = views[0]->getTangent(perm[0]);
	t2_1 = views[0]->getTangent(perm[1]);
	t3_1 = views[0]->getTangent(perm[2]);
	
	t1_2 = views[1]->getTangent(perm[0]);
	t2_2 = views[1]->getTangent(perm[1]);
	t3_2 = views[1]->getTangent(perm[2]);

	t1_3 = views[2]->getTangent(perm[0]);
	t2_3 = views[2]->getTangent(perm[1]);
	t3_3 = views[2]->getTangent(perm[2]);

	cout << p1_1[0] << " " << p1_1[1] << endl;
	cout << p2_1[0] << " " << p2_1[1] << endl;
	cout << p3_1[0] << " " << p3_1[1] << endl;
	cout << endl;
	cout << p1_2[0] << " " << p1_2[1] << endl;
	cout << p2_2[0] << " " << p2_2[1] << endl;
	cout << p3_2[0] << " " << p3_2[1] << endl;
	cout << endl;
	cout << p1_3[0] << " " << p1_3[1] << endl;
	cout << p2_3[0] << " " << p2_3[1] << endl;
	cout << p3_3[0] << " " << p3_3[1] << endl;
	cout << endl;

	cout << t1_1[0] << " " << t1_1[1] << endl;
	cout << t2_1[0] << " " << t2_1[1] << endl;
	cout << t3_1[0] << " " << t3_1[1] << endl;
	cout << endl;
	cout << t1_2[0] << " " << t1_2[1] << endl;
	cout << t2_2[0] << " " << t2_2[1] << endl;
	cout << t3_2[0] << " " << t3_2[1] << endl;
	cout << endl;
	cout << t1_3[0] << " " << t1_3[1] << endl;
	cout << t2_3[0] << " " << t2_3[1] << endl;
	cout << t3_3[0] << " " << t3_3[1] << endl;
	cout << endl;


	cout <<  "0 1" << endl;
	cout << endl;
	cout << views[0]->getK()[0] << " " << views[0]->getK()[1] << " " << views[0]->getK()[2] << endl;
	cout << views[0]->getK()[3] << " " << views[0]->getK()[4] << " " << views[0]->getK()[5] << endl;
	cout << endl;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << R1[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << C1[0][0] << " " << C1[1][0] << " " << C1[2][0] << endl;
	cout << endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << R2[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << C2[0][0] << " " << C2[1][0] << " " << C2[2][0] << endl;
	cout << endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << R3[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << C3[0][0] << " " << C3[1][0] << " " << C3[2][0] << endl;
	cout << endl;

} 

void generateParameters_5pt2Views16Unknowns(vector<singleView*> views, double* parameter, double* groundTruth)
{
	double (*R1)[3] = (double (*)[3])&views[0]->getR()[0];
	double (*R2)[3] = (double (*)[3])&views[1]->getR()[0];
	double R12[3][3];
	double T12[3];
	double C1_C2[3];

	for (int i = 0; i < 9; i++) {
		R12[i/3][i%3] = 0.0;
	}

	for (int i = 0; i < 3; i++) {
		T12[i] = 0.0;
		C1_C2[i] = 0.0;
	}

	double R1_inv[3][3], R2_inv[3][3];
	matTranspose(R1,R1_inv,3,3);
	matTranspose(R2,R2_inv,3,3);

	RRMulti(R2,R1_inv, R12, 3,3,3,3);
	
	vecSubs(views[0]->getC(),views[1]->getC(),C1_C2,3,1);
	//cout << endl;
	//cout << views[0]->getC()[0] << " " << views[0]->getC()[1] << " " << views[0]->getC()[2] << endl;
	//cout << views[1]->getC()[0] << " " << views[1]->getC()[1] << " " << views[1]->getC()[2] << endl;
	//cout << endl;

	RTMulti(R2,C1_C2,T12, 3, 3, 3, 1);
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			cout << R1[i][j] << " ";
//		}
//		cout << endl;
//	}

//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			cout << R2[i][j] << " ";
//		}
//		cout << endl;
//	}

//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//		{
//			cout << R12[i][j] << " ";
//		}
//		cout << endl;
//	}
	double R1_v[9];
	for (int i = 0; i < 9; i++)
	{
		R1_v[i] = R12[i / 3][i % 3];
	}

	rotm2quat(R1_v, groundTruth);
	for (int i = 0; i < 2; i++)
	{
		groundTruth[4+i] = T12[i] / T12[2];
	}

//	for (int i = 0; i < 16; i++)
//	{
//		cout << groundTruth[i] << endl;
//	}

	//Get Points

	vector<int> perm(views[0]->getNum());
	for (int i = 0; i < views[0]->getNum(); i++)
	{
		perm[i] = i;
	}

	random_shuffle(perm.begin(), perm.end());
//	perm[0] = 10;
//	perm[1] = 20;
//	perm[2] = 30;
	double *p1_1, *p2_1, *p3_1, *p4_1, *p5_1;
	double *p1_2, *p2_2, *p3_2, *p4_2, *p5_2;


	p1_1 = views[0]->getPoint(perm[0]);
	p2_1 = views[0]->getPoint(perm[1]);
	p3_1 = views[0]->getPoint(perm[2]);
	p4_1 = views[0]->getPoint(perm[3]);
	p5_1 = views[0]->getPoint(perm[4]);

	p1_2 = views[1]->getPoint(perm[0]);
	p2_2 = views[1]->getPoint(perm[1]);
	p3_2 = views[1]->getPoint(perm[2]);
	p4_2 = views[1]->getPoint(perm[3]);
	p5_2 = views[1]->getPoint(perm[4]);

//	cout << endl;
//	cout << p1_1[0] << " " << p1_1[1] << endl;	
//	cout << endl;	
	parameter[0] = (1/views[0]->getK()[0]) * p1_1[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[1] = (1/views[0]->getK()[4]) * p1_1[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[2] = (1/views[0]->getK()[0]) * p2_1[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[3] = (1/views[0]->getK()[4]) * p2_1[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[4] = (1/views[0]->getK()[0]) * p3_1[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[5] = (1/views[0]->getK()[4]) * p3_1[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[6] = (1/views[0]->getK()[0]) * p4_1[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[7] = (1/views[0]->getK()[4]) * p4_1[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[8] = (1/views[0]->getK()[0]) * p5_1[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[9] = (1/views[0]->getK()[4]) * p5_1[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	

	parameter[10] = (1/views[0]->getK()[0]) * p1_2[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[11] = (1/views[0]->getK()[4]) * p1_2[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[12] = (1/views[0]->getK()[0]) * p2_2[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[13] = (1/views[0]->getK()[4]) * p2_2[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[14] = (1/views[0]->getK()[0]) * p3_2[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[15] = (1/views[0]->getK()[4]) * p3_2[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[16] = (1/views[0]->getK()[0]) * p4_2[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[17] = (1/views[0]->getK()[4]) * p4_2[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	
	parameter[18] = (1/views[0]->getK()[0]) * p5_2[0] - (views[0]->getK()[2] / views[0]->getK()[0]);
	parameter[19] = (1/views[0]->getK()[4]) * p5_2[1] - (views[0]->getK()[5] / views[0]->getK()[4]);	

}


void generateParameters_chicago(vector<singleView*> views, double* parameters, double* expected_solution)
{
	//Compute 14 Ground Truth
	//R12, R13, R23, T12, T13, T23	
	double (*R1)[3] = (double (*)[3])&views[0]->getR()[0];
	double (*R2)[3] = (double (*)[3])&views[1]->getR()[0];
	double (*R3)[3] = (double (*)[3])&views[2]->getR()[0];

	double (*C1)[1] = (double (*)[1])&views[0]->getC()[0];
	double (*C2)[1] = (double (*)[1])&views[1]->getC()[0];
	double (*C3)[1] = (double (*)[1])&views[2]->getC()[0];
	

	double R12[3][3], R23[3][3], R13[3][3];
	double T12[3], T23[3], T13[3];
	double C1_C2[3];
	double C2_C3[3];
	
	for (int i = 0; i < 9; i++) {
		R12[i/3][i%3] = 0.0;
		R23[i/3][i%3] = 0.0;
		R13[i/3][i%3] = 0.0;
	}

	for (int i = 0; i < 3; i++) {
		T12[i] = 0.0;
		T23[i] = 0.0;
		T13[i] = 0.0;
		C1_C2[i] = 0.0;
		C2_C3[i] = 0.0;
	}


	double R1_inv[3][3], R2_inv[3][3], R3_inv[3][3];
	matTranspose(R1,R1_inv,3,3);
	matTranspose(R2,R2_inv,3,3);
	matTranspose(R3,R3_inv,3,3);

	RRMulti(R2,R1_inv, R12, 3,3,3,3);
	RRMulti(R3,R2_inv, R23, 3,3,3,3);
	RRMulti(R23,R12,R13,3,3,3,3);
	
	vecSubs(views[0]->getC(),views[1]->getC(),C1_C2,3,1);
	RTMulti(R2,C1_C2,T12, 3, 3, 3, 1);
	vecSubs(views[1]->getC(),views[2]->getC(),C2_C3,3,1);
	RTMulti(R3,C2_C3,T23,3,3,3,1);

	
	double T_temp[3];
	RTMulti(R23,T12,T_temp,3,3,3,1);
	vecAdd(T_temp,T23,T13,3,1);
	//Generate Expected Solution
	double R1_v[9];
	double R2_v[9];
	for (int i = 0; i < 9; i++)
	{
		R1_v[i] = R12[i / 3][i % 3];
		R2_v[i] = R13[i / 3][i % 3];
	}
	rotm2quat(R1_v, expected_solution);
	rotm2quat(R2_v, expected_solution + 4);
	for (int i = 0; i < 3 ; i++)
	{
		expected_solution[8+i] = T12[i];
		expected_solution[8+3+i] = T13[i];
	}
//	
//	
//	cout << "//////////////////////////////////////////////" << endl;
//	//Compute 56 Parameters
	//Randomly select 3 points
	vector<int> perm(views[0]->getNum());
	for (int i = 0; i < views[0]->getNum(); i++)
	{
		perm[i] = i;
	}
	random_shuffle(perm.begin(), perm.end());
//	
//
	//Test Code
	perm[0] = 0;
	perm[1] = 49;
	perm[2] = 149;
//
	cout << perm[0] << " " << perm[1] << " " << perm[2] << endl;	
	double *p1_1, *p2_1, *p3_1, *p1_2, *p2_2, *p3_2, *p1_3, *p2_3, *p3_3;
	double *t1_1, *t2_1, *t3_1, *t1_2, *t2_2, *t3_2, *t1_3, *t2_3, *t3_3;
	p1_1 = views[0]->getPoint(perm[0]);
	p2_1 = views[0]->getPoint(perm[1]);
	p3_1 = views[0]->getPoint(perm[2]);
	
	p1_2 = views[1]->getPoint(perm[0]);
	p2_2 = views[1]->getPoint(perm[1]);
	p3_2 = views[1]->getPoint(perm[2]);

	p1_3 = views[2]->getPoint(perm[0]);
	p2_3 = views[2]->getPoint(perm[1]);
	p3_3 = views[2]->getPoint(perm[2]);

	t1_1 = views[0]->getTangent(perm[0]);
	t2_1 = views[0]->getTangent(perm[1]);
	t3_1 = views[0]->getTangent(perm[2]);
	
	t1_2 = views[1]->getTangent(perm[0]);
	t2_2 = views[1]->getTangent(perm[1]);
	t3_2 = views[1]->getTangent(perm[2]);

	t1_3 = views[2]->getTangent(perm[0]);
	t2_3 = views[2]->getTangent(perm[1]);
	t3_3 = views[2]->getTangent(perm[2]);
	
	//Lines and Chart
	double al1[3][5], al2[3][5], al3[3][5];
	double *al1_pp[3], *al2_pp[3], *al3_pp[3];
	for (int i = 0; i < 3; ++i)
	{
    	al1_pp[i] = al1[i];
		al2_pp[i] = al2[i];
		al3_pp[i] = al3[i];
	}
	
	double **al1_p = al1_pp, **al2_p = al2_pp, **al3_p = al3_pp;
	
	getLines(p1_1,p2_1,p3_1,t1_1,t2_1,t3_1,views[0]->getK(),al1_p);
	getLines(p1_2,p2_2,p3_2,t1_2,t2_2,t3_2,views[1]->getK(),al2_p);
	getLines(p1_3,p2_3,p3_3,t1_3,t2_3,t3_3,views[2]->getK(),al3_p);

	cout << endl;

	//for (int i = 0; i < 3; i++)
	//{
	//	for (int j = 0; j < 5; j++)
	//	{
	//		cout << al1_p[i][j] << " ";
	//	}
	//	cout << endl;

	//}
	

	double visibleLines[15][3];
	//
	
	int counter = 0;
	for (int i = 0; i < 5; i++)
	{
		visibleLines[counter][0] = al1[0][i];
		visibleLines[counter][1] = al1[1][i];
		visibleLines[counter][2] = al1[2][i];
		counter++;
		visibleLines[counter][0] = al2[0][i];
		visibleLines[counter][1] = al2[1][i];
		visibleLines[counter][2] = al2[2][i];
		counter++;
		visibleLines[counter][0] = al3[0][i];
		visibleLines[counter][1] = al3[1][i];
		visibleLines[counter][2] = al3[2][i];
		counter++;


	}

	double *vl_pp[15];
	for (int i = 0; i < 15; i++)
	{
		vl_pp[i] = visibleLines[i];
	}
	double** vl_p = vl_pp;

	cout << "--------visibleLines------------" << endl;
	cout << endl;
	for (int i = 0; i < 15; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << visibleLines[i][j] << " ";
		}
		cout << endl;
	}
	cout << "----------------------------------" << endl;
	cout << "---------vl_p-------------" << endl;

	cout << endl;
	for (int i = 0; i < 15; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << vl_p[i][j] << " ";
		}
		cout << endl;
	}
//	int counter = 0;
//	for (int i = 0; i < 5; i++)
//	{
//		visibleLines.row(counter++) = al1.col(i).transpose();
//		visibleLines.row(counter++) = al2.col(i).transpose();
//		visibleLines.row(counter++) = al3.col(i).transpose();
//	}
	cout << "----------------------------------" << endl;

	//Put Chart in:
	lines2Params(vl_p, parameters);
//	cout << endl;	
//	for (int i = 0; i < 56; i++)
//	{
//		cout << parameters[i] << endl;
//	}
	
};


//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R1[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R2[i][j] << " ";
//		cout << endl;
//	}
//
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R3[i][j] << " ";
//		cout << endl;
//	}
//	cout << endl;
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R12[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R23[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R13[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		cout << T12[i] << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		cout << T23[i] << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		cout << T13[i] << endl;
//	}
//	double* R23 = views[2].getR() * views[1].getR().inverse();
//	double* T12 = views[1].getR() * (views[0].getC() - views[1].getC());
//	double* T23 = views[2].getR() * (views[1].getC() - views[2].getC());
//	double* R13 = R23 * R12;
//	double* T13 = R23 * T12 + T23;

//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R1[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R2[i][j] << " ";
//		cout << endl;
//	}
//
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R3[i][j] << " ";
//		cout << endl;
//	}
//	cout << endl;
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R12[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R23[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			cout << R13[i][j] << " ";
//		cout << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		cout << T12[i] << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		cout << T23[i] << endl;
//	}
//	for (int i = 0; i < 3; i++)
//	{
//		cout << T13[i] << endl;
//	}
//	double* R23 = views[2].getR() * views[1].getR().inverse();
//	double* T12 = views[1].getR() * (views[0].getC() - views[1].getC());
//	double* T23 = views[2].getR() * (views[1].getC() - views[2].getC());
//	double* R13 = R23 * R12;
//	double* T13 = R23 * T12 + T23;

