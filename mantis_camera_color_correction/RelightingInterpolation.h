/**
* @brief C++ head file of class RelightingInterpolation
* use edge aware interpolator to realize local relighting
* @author: Shane Yuan
* @date: oct 8, 2016
*/

#ifndef RELIGHTING_INTERPOLATION_H
#define RELIGHTING_INTERPOLATION_H

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 

#include "EdgeAwareInterpolator.h"

// add definition of vec12f
namespace cv {
	typedef Vec<float, 12> Vec12f;
}

// color tranform class used as template
class ColorTrans {
private:
	
public:
	float trans[12];
	// overload operator function
	ColorTrans() {
		for (int i = 0; i < 12; i ++) {
			trans[i] = 0;
		}
	}
	ColorTrans(cv::Vec12f val) {
		for (int i = 0; i < 12; i++) {
			trans[i] = val.val[i];
		}
	}
	ColorTrans(Eigen::MatrixXf transformR, Eigen::MatrixXf transformT) {
		trans[0] = transformR(0, 0); trans[1] = transformR(0, 1); trans[2] = transformR(0, 2); trans[9] = transformT(0) * 255;
		trans[3] = transformR(1, 0); trans[4] = transformR(1, 1); trans[5] = transformR(1, 2); trans[10] = transformT(1) * 255;
		trans[6] = transformR(2, 0); trans[7] = transformR(2, 1); trans[8] = transformR(2, 2); trans[11] = transformT(2) * 255;
	}
	friend ColorTrans operator+(const ColorTrans cL, const ColorTrans cR) {
		ColorTrans c;
		for (int i = 0; i < 12; i++) {
			c.trans[i] = cL.trans[i] + cR.trans[i];
		}
		return c;
	}
	ColorTrans operator *(float a) {
		ColorTrans c;
		for (int i = 0; i < 12; i++) {
			c.trans[i] = trans[i] * a;
		}
		return c;
	}
	// change to cv::Vec12f
	cv::Vec12f toVec() {
		cv::Vec12f val;
		for (int i = 0; i < 12; i ++) {
			val.val[i] = trans[i];
		}
		return val;
	}
	// generate transform matrix
	cv::Mat genTransformMat() {
		cv::Mat_<float> transformMat = cv::Mat::zeros(3, 4, CV_32F);
		transformMat(0, 0) = trans[0]; transformMat(0, 1) = trans[1]; transformMat(0, 2) = trans[2]; transformMat(0, 3) = trans[9];
		transformMat(1, 0) = trans[3]; transformMat(1, 1) = trans[4]; transformMat(1, 2) = trans[5]; transformMat(1, 3) = trans[10];
		transformMat(2, 0) = trans[6]; transformMat(2, 1) = trans[7]; transformMat(2, 2) = trans[8]; transformMat(2, 3) = trans[11];
		return transformMat;
	}
	// apply transform to pixel values
	cv::Vec3b apply(cv::Vec3b v) {
		float b = static_cast<float>(v.val[0]);
		float g = static_cast<float>(v.val[1]);
		float r = static_cast<float>(v.val[2]);
		float newB = trans[0] * b + trans[1] * g + trans[2] * r + trans[9];
		float newG = trans[3] * b + trans[4] * g + trans[5] * r + trans[10];
		float newR = trans[6] * b + trans[7] * g + trans[8] * r + trans[11];
		newB = std::max(std::min(newB, 255.0f), 0.0f);
		newG = std::max(std::min(newG, 255.0f), 0.0f);
		newR = std::max(std::min(newR, 255.0f), 0.0f);
		return cv::Vec3b(static_cast<uchar>(newB), static_cast<uchar>(newG), static_cast<uchar>(newR));
	}
	// apply transform to whole matrix
	cv::Mat apply(cv::Mat input) {
		cv::Mat output;
		transform(input, output, genTransformMat());
		return output;
	}
};

typedef struct struct_Superpixel {
	std::vector<cv::Vec3b> pVal;
	std::vector<cv::Point2i> pPos;
	cv::Point2f center;
	bool update;
	struct_Superpixel() {
		pVal.clear(); pPos.clear(); center = cv::Point2f(-1, -1); update = true;
	}
	// clear
	int clear() {
		pVal.clear(); pPos.clear(); center = cv::Point2f(-1, -1); update = true;
	}
	// add pixels
	int addPixel(cv::Vec3b val, cv::Point2i pos) {
		pVal.push_back(val);
		pPos.push_back(pos);
		update = false;
		return 0;
	}
	// update center
	cv::Point2f getCenter() {
		if (update == false) {
			center = cv::Point2f(0, 0);
			for (int i = 0; i < pPos.size(); i ++) {
				center += cv::Point2f(pPos[i].x, pPos[i].y);
			}
			center = center / static_cast<float>(pPos.size());
			update = true;
		}
		return center;
	}
} Superpixel;

class MKLtransform {
private:
	cv::Mat src;
	cv::Mat dst;
	Superpixel spSrc;
	Superpixel spDst;
	bool isSuperpixel;
public:

private:

protected:
	// reshape matrices
	// reshape between OpenCV MxNx3 matrix and Eigen (MN) x 3 matrix (doesn't consider black pixels)
	Eigen::MatrixXf toEigenMat(cv::Mat input);
	cv::Mat toOpenCVMat(Eigen::MatrixXf input, int width, int height);
	// reshape between OpenCV MxNx3 matrix and Eigen (MN - ?) x 3 matrix (discard black pixels)
	int toEigenMat(cv::Mat refMat, cv::Mat detailMat, Eigen::MatrixXf & refEigen, Eigen::MatrixXf & detailEigen, cv::Mat mask);
	int toOpenCVMat(cv::Mat refMat, cv::Mat & detailMat, Eigen::MatrixXf refEigen, Eigen::MatrixXf detailEigen, cv::Mat mask);
	// reshape between superpixel and eigen matrix
	Eigen::MatrixXf toEigenMat(Superpixel sp);
	// truncate pixel value to 0 to 1
	int truncatePixelVal(Eigen::MatrixXf input, Eigen::MatrixXf& output);

	// Monge-Kantorovich linear transformation
	// compute mean vector, zero-mean data and co-variance matrix
	int calcMeanCov(Eigen::MatrixXf input, Eigen::MatrixXf & meanVec, Eigen::MatrixXf & zeroMeanMat, Eigen::MatrixXf & covMat);
	// calculate Monge-Kantorovich linear transform matrix
	int calcMKLtransform(Eigen::MatrixXf covMatSrc, Eigen::MatrixXf covMatTar, Eigen::MatrixXf & transformMat);
	// change eigen value matrix positive
	Eigen::MatrixXf postiveEigenValues(Eigen::MatrixXf eigenValueMat);
	// inverse diagonal matrix
	Eigen::MatrixXf invDiagMat(Eigen::MatrixXf diagMat);

public:
	MKLtransform();
	MKLtransform(cv::Mat src, cv::Mat dst);
	~MKLtransform();
	// set input
	int setInput(cv::Mat src, cv::Mat dst);
	int setInput(Superpixel spSrc, Superpixel spDst);
	// mkl transform
	ColorTrans estimateMKLTransform(int useMask = 0);
	ColorTrans estimateMKLTransform(cv::Mat mask);

	// utility function
	// compute mask
	static cv::Mat imfillholes(cv::Mat src);
	static cv::Mat computeMask(cv::Mat & img, float thresh = 75);
	static cv::Mat computeMask(cv::Mat & img, float threshDark, float threshBright);
	static cv::Mat fusionMask(cv::Mat mask1, cv::Mat mask2);
	static cv::Mat applyMask(cv::Mat & img, cv::Mat & mask);
};

class RelightingInterpolation {
private:
	cv::Mat src;
	cv::Mat dst;
	std::vector<ColorTrans> transform;
	std::vector<cv::Point2f> samples;
	cv::Mat mask;
	cv::Mat srcMask;
public:

private:
	// check if the rectangle is valid in mask
	bool checkMask(cv::Rect rect);
public:
	RelightingInterpolation();
	~RelightingInterpolation();

	// set input
	int setInput(cv::Mat src, cv::Mat dst);
	int setInput(cv::Mat src, cv::Mat srcMask, cv::Mat dst, cv::Mat dstMask);
	// interpolate relighting color transform matrices
	int interpolate(std::vector<cv::Point2f> samples, std::vector<ColorTrans> colorTrans, std::vector<ColorTrans> & colorTransInter, cv::Mat & labels);
	// relighting
	cv::Mat relighting(std::vector<ColorTrans> colorTransInter, cv::Mat labels, int filterPara = 0);


	// generate regular samples using regular patches
	int genRegularSamples(int step, int patchSize, std::vector<cv::Point2f> & samples, std::vector<ColorTrans> & colorTrans);
	// generate content based samples using superpixels
	int genSuperPixelSamples(int & labelNum, std::vector<cv::Point2f> & samples, std::vector<ColorTrans> & colorTrans);

	// paper testing only
	int interpolateBilinear(int step, int patchSize, std::vector<cv::Point2f> samples, std::vector<ColorTrans> colorTrans, cv::Mat inputImg, cv::Mat & outImg);
	
};

#endif
