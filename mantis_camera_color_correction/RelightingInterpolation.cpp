/**
* @brief C++ sourcee file of class RelightingInterpolation
* use edge aware interpolator to realize local relighting
* @author: Shane Yuan
* @date: oct 8, 2016
*/

#include <opencv2/ximgproc.hpp>
#include "RelightingInterpolation.h"

#define RELIGHTING_MAX_PIXEL_VALUE 255.0
#define RELIGHTING_EPS 2.2204e-16

// mkl transform estimation class
MKLtransform::MKLtransform() {};
MKLtransform::~MKLtransform() {};
MKLtransform::MKLtransform(cv::Mat src, cv::Mat dst) {
	this->setInput(src, dst);
}
int MKLtransform::setInput(cv::Mat src, cv::Mat dst) {
	this->src = src;
	this->dst = dst;
	this->isSuperpixel = false;
	return ErrorCode::RIGHT;
}
int MKLtransform::setInput(Superpixel spSrc, Superpixel spDst) {
	this->spSrc = spSrc;
	this->spDst = spDst;
	this->isSuperpixel = true;
	return ErrorCode::RIGHT;
}

// utility function
/**
* @brief fill hole in mask image
* @param cv::Mat src: input mask image with holes
* @return cv::Mat: return mask image after filling holes
*/
cv::Mat MKLtransform::imfillholes(cv::Mat src) {
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	if (!contours.empty() && !hierarchy.empty()) {
		for (int idx = 0; idx < contours.size(); idx++) {
			drawContours(src, contours, idx, cv::Scalar::all(255), CV_FILLED, 8);
		}
	}
	return src;
}

/**
* @brief compute mask of image (because warped detail images have black background regions)
* @param cv::Mat & img: input image
* @return cv::Mat: mask map
*/
cv::Mat MKLtransform::computeMask(cv::Mat & img, float thresh /* = 75 */) {
	cv::Mat mask(img.rows, img.cols, CV_8U, 255);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			cv::Vec3b val = img.at<cv::Vec3b>(i, j);
			int valTot = val.val[0] + val.val[1] + val.val[2];
			if (valTot < thresh) {
				mask.at<uchar>(i, j) = 0;
			}
		}
	}
	mask = MKLtransform::imfillholes(mask);
	return mask;
}

/**
* @brief compute mask of image in relighting (too dark or too bright pixels will be discard)
* @param cv::Mat & img: input image
* @return cv::Mat: mask map
*/
cv::Mat MKLtransform::computeMask(cv::Mat & img, float threshDark, float threshBright) {
	cv::Mat mask(img.rows, img.cols, CV_8U, 255);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			cv::Vec3b val = img.at<cv::Vec3b>(i, j);
			int valTot = static_cast<int>(val.val[0]) + static_cast<int>(val.val[1])
				+ static_cast<int>(val.val[2]);
			if (valTot < threshDark) {
				mask.at<uchar>(i, j) = 0;
			} 
			else if (valTot > threshBright) {
				mask.at<uchar>(i, j) = 0;
			}
		}
	}
//	mask = MKLtransform::imfillholes(mask);
	return mask;
}

/**
* @brief merge two masks
* @param cv::Mat mask1: input mask1
* @param cv::Mat mask2: input mask2
* @return cv::Mat: output mask map using the smaller value
*/
cv::Mat MKLtransform::fusionMask(cv::Mat mask1, cv::Mat mask2) {
	cv::Mat mask(mask1.rows, mask1.cols, CV_8U, 255);
	for (int i = 0; i < mask1.rows; i++) {
		for (int j = 0; j < mask1.cols; j++) {
			uchar val1 = mask1.at<uchar>(i, j);
			uchar val2 = mask2.at<uchar>(i, j);
			mask.at<uchar>(i, j) = std::min(val1, val2);
		}
	}
	return mask;
}

/**
* @brief apply mask to an input image
* @param cv::Mat img: input image
* @param cv::Mat mask: input mask
* @return cv::Mat: output image
*/
cv::Mat MKLtransform::applyMask(cv::Mat & img, cv::Mat & mask) {
	cv::Mat output = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (mask.at<uchar>(i, j) == 0) {
				output.at<cv::Vec3b>(i, j) = 0;
			}
		}
	}
	return output;
}

/**
* @brief reshape OpenCV MxNx3 matrix to Eigen (MN) x 3 matrix
* @praram cv::Mat input: input OpenCV matrix need to transfer
* @return Eigen::MatrixXf: output corresponding eigen matrix
*/
Eigen::MatrixXf MKLtransform::toEigenMat(cv::Mat input) {
	Eigen::MatrixXf output;
	int imgWidth = input.cols;
	int imgHeight = input.rows;
	int len = imgWidth * imgHeight;
	output.resize(len, 3);
	int index = 0;
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			cv::Vec3b val = input.at<cv::Vec3b>(i, j);
			output(index, 0) = static_cast<float>(val.val[0]) / RELIGHTING_MAX_PIXEL_VALUE;
			output(index, 1) = static_cast<float>(val.val[1]) / RELIGHTING_MAX_PIXEL_VALUE;
			output(index, 2) = static_cast<float>(val.val[2]) / RELIGHTING_MAX_PIXEL_VALUE;
			index++;
		}
	}
	return output;
}

/**
* @brief reshape Superpixel data to Eigen (MN) x 3 matrix
* @praram Superpixel sp: superpixel data
* @return Eigen::MatrixXf: output corresponding eigen matrix
*/
Eigen::MatrixXf MKLtransform::toEigenMat(Superpixel sp) {
	Eigen::MatrixXf output;
	int len = sp.pVal.size();
	output.resize(len, 3);
	int index = 0;
	for (int i = 0; i < len; i++) {
		cv::Vec3b val = sp.pVal[i];
		output(index, 0) = static_cast<float>(val.val[0]) / RELIGHTING_MAX_PIXEL_VALUE;
		output(index, 1) = static_cast<float>(val.val[1]) / RELIGHTING_MAX_PIXEL_VALUE;
		output(index, 2) = static_cast<float>(val.val[2]) / RELIGHTING_MAX_PIXEL_VALUE;
		index++;
	}
	return output;
}

/**
* @brief reshape OpenCV MxNx3 matrix to Eigen (MN) x 3 matrix
* @param Eigen::MatrixXf input: input eigen matrix need to transfer
* @param int width: original image width
* @param int height: original image height
* @return cv::Mat: output corresponding OpenCV matrix
*/
cv::Mat MKLtransform::toOpenCVMat(Eigen::MatrixXf input, int width, int height) {
	cv::Mat output;
	output = cv::Mat::zeros(height, width, CV_8UC3);
	int len = input.rows();
	if (len != height * width) {
		std::cerr << "input and output image matrix size not match ! return zero matrix!" << std::endl;
		return cv::Mat::zeros(height, width, CV_8UC3);
	}
	int index = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cv::Vec3b val;
			val.val[0] = static_cast<uchar>(input(index, 0) * RELIGHTING_MAX_PIXEL_VALUE);
			val.val[1] = static_cast<uchar>(input(index, 1) * RELIGHTING_MAX_PIXEL_VALUE);
			val.val[2] = static_cast<uchar>(input(index, 2) * RELIGHTING_MAX_PIXEL_VALUE);
			output.at<cv::Vec3b>(i, j) = val;
			index++;
		}
	}
	return output;
}

/**
* @brief reshape OpenCV MxNx3 matrix to Eigen (MN) x 3 matrix
* @param cv::Mat refMat: reference image OpenCV matrix
* @param cv::Mat detailMat: high resolution image OpenCV matrix
* @param Eigen::MatrixXf & refEigen: corresponding reference eigen matrix
* @param Eigen::MatrixXf & detailEigen: corresponding high resolution eigen matrix
* @param cv::Mat mask: mask matrix denotes the if the pixel is a black pixel
* @return cv::Mat: output corresponding OpenCV matrix
*/
int MKLtransform::toEigenMat(cv::Mat refMat, cv::Mat detailMat, Eigen::MatrixXf& refEigen, Eigen::MatrixXf& detailEigen, cv::Mat mask) {
	int imgWidth = refMat.cols;
	int imgHeight = refMat.rows;
	int len = imgWidth * imgHeight;
	refEigen.resize(len, 3);
	detailEigen.resize(len, 3);
	int index = 0;
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			cv::Vec3b detailVal = detailMat.at<cv::Vec3b>(i, j);
			if (mask.at<uchar>(i, j) == 255) {
				detailEigen(index, 0) = static_cast<float>(detailVal.val[0]) / RELIGHTING_MAX_PIXEL_VALUE;
				detailEigen(index, 1) = static_cast<float>(detailVal.val[1]) / RELIGHTING_MAX_PIXEL_VALUE;
				detailEigen(index, 2) = static_cast<float>(detailVal.val[2]) / RELIGHTING_MAX_PIXEL_VALUE;
				cv::Vec3b refVal = refMat.at<cv::Vec3b>(i, j);
				refEigen(index, 0) = static_cast<float>(refVal.val[0]) / RELIGHTING_MAX_PIXEL_VALUE;
				refEigen(index, 1) = static_cast<float>(refVal.val[1]) / RELIGHTING_MAX_PIXEL_VALUE;
				refEigen(index, 2) = static_cast<float>(refVal.val[2]) / RELIGHTING_MAX_PIXEL_VALUE;
				index++;
			}
		}
	}
	refEigen.conservativeResize(index, 3);
	detailEigen.conservativeResize(index, 3);
	return ErrorCode::RIGHT;
}

/**
* @brief reshape OpenCV MxNx3 matrix to Eigen (MN) x 3 matrix
* @param cv::Mat refMat: reference image OpenCV matrix
* @param cv::Mat & detailMat: high resolution image OpenCV matrix
* @param Eigen::MatrixXf refEigen: corresponding reference eigen matrix
* @param Eigen::MatrixXf detailEigen: corresponding high resolution eigen matrix
* @param cv::Mat mask: mask matrix denotes the if the pixel is a black pixel
* @return cv::Mat: output corresponding OpenCV matrix
*/
int MKLtransform::toOpenCVMat(cv::Mat refMat, cv::Mat & detailMat, Eigen::MatrixXf refEigen, Eigen::MatrixXf detailEigen, cv::Mat mask) {
	int imgWidth = refMat.cols;
	int imgHeight = refMat.rows;
	int index = 0;
	for (int i = 0; i < imgHeight; i++) {
		for (int j = 0; j < imgWidth; j++) {
			cv::Vec3b detailVal = detailMat.at<cv::Vec3b>(i, j);
			if (mask.at<uchar>(i, j) == 255) {
				cv::Vec3b val;
				val.val[0] = static_cast<uchar>(detailEigen(index, 0) * RELIGHTING_MAX_PIXEL_VALUE);
				val.val[1] = static_cast<uchar>(detailEigen(index, 1) * RELIGHTING_MAX_PIXEL_VALUE);
				val.val[2] = static_cast<uchar>(detailEigen(index, 2) * RELIGHTING_MAX_PIXEL_VALUE);
				detailMat.at<cv::Vec3b>(i, j) = val;
				index++;
			}
			else {
				detailMat.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
		}
	}
	return ErrorCode::RIGHT;
}

/**
* @breif reshape OpenCV MxNx3 matrix to Eigen (MN) x 3 matrix
* @param Eigen::MatrixXf input: input eigen matrix need to truncate pixel value
* @param Eigen::MatrixXf& output: output eigen matrix whose pixel values are truncated
* @return int: ErrorCode
*/
int MKLtransform::truncatePixelVal(Eigen::MatrixXf input, Eigen::MatrixXf& output) {
	int len = input.rows();
	output.resize(len, 3);
	for (int i = 0; i < len; i++) {
		for (int j = 0; j < 3; j++) {
			float val = input(i, j);
			if (val < 0) {
				val = 0;
			}
			else if (val > 1) {
				val = 1;
			}
			output(i, j) = val;
		}
	}
	return ErrorCode::RIGHT;
}


/**
* @brief compute mean vector, zero-mean data and co-variance matrix
* @param Eigen::MatrixXf input: input (MN) x 3 data matrix which records all the pixel values of the input image
* @param Eigen::MatrixXf & meanVec: mean vector of the data, mean color of image pixels
* @param Eigen::MatrixXf & zeroMeanMat: (MN) x 3 data matrix after substract the mean vector
* @param Eigen::MatrixXf & covMat: co-variance matrix
* @return int: ErrorCode
*/
int MKLtransform::calcMeanCov(Eigen::MatrixXf input, Eigen::MatrixXf & meanVec, Eigen::MatrixXf & zeroMeanMat, Eigen::MatrixXf & covMat) {
	// calculate mean vector
	meanVec = input.colwise().mean();
	// calculate zero mean data
	zeroMeanMat = input;
	Eigen::RowVectorXf meanVecRow(Eigen::RowVectorXf::Map(meanVec.data(), 3));
	zeroMeanMat.rowwise() -= meanVecRow;
	// calculate covariance matrix
	covMat = (zeroMeanMat.adjoint() * zeroMeanMat) / double(input.rows() - 1);
#ifdef MY_LOG
#ifdef MY_DEBUG
	MyUtility::dumpLog(meanVecRow);
	MyUtility::dumpLog(covMat);
#endif
#endif
	return ErrorCode::RIGHT;
}

/**
* @brief calculate mkl transform
* @param Eigen::MatrixXf covMatSrc: input source co-variance matrix
* @param Eigen::MatrixXf covMatTar: input target co-variance matrix
* @param Eigen::MatrixXf & transformMat: output 3x3 color transfor matrix
* @return int: ErrorCode
*/
int MKLtransform::calcMKLtransform(Eigen::MatrixXf covMatSrc, Eigen::MatrixXf covMatTar, Eigen::MatrixXf & transformMat) {
	// eigen decomposition of source matrix
	Eigen::EigenSolver<Eigen::MatrixXf> eigSrc(covMatSrc);
	Eigen::MatrixXf eigSrcVal = eigSrc.pseudoEigenvalueMatrix();
	Eigen::MatrixXf eigSrcVec = eigSrc.pseudoEigenvectors();
	eigSrcVal = postiveEigenValues(eigSrcVal);
#ifdef MY_LOG
#ifdef MY_DEBUG
	MyUtility::dumpLog("Dump src eigen matrix ......\n");
	MyUtility::dumpLog(eigSrcVal);
	MyUtility::dumpLog(eigSrcVec);
	MyUtility::dumpLog("\n");
#endif
#endif
	// compute C matrix
	Eigen::MatrixXf eigSqrtSrcVal = eigSrcVal.cwiseSqrt();
	Eigen::MatrixXf C = eigSqrtSrcVal * eigSrcVec.transpose() * covMatTar * eigSrcVec * eigSqrtSrcVal;
#ifdef MY_LOG
#ifdef MY_DEBUG
	MyUtility::dumpLog("Dump C matrix ......\n");
	MyUtility::dumpLog(C);
	MyUtility::dumpLog("\n");
#endif
#endif
	// eigen decomposition of C matrix
	Eigen::EigenSolver<Eigen::MatrixXf> eigC(C);
	Eigen::MatrixXf eigCVal = eigC.pseudoEigenvalueMatrix();
	Eigen::MatrixXf eigCVec = eigC.pseudoEigenvectors();
	eigCVal = postiveEigenValues(eigCVal);
#ifdef MY_LOG
#ifdef MY_DEBUG
	MyUtility::dumpLog("Dump C eigen matrix ......\n");
	MyUtility::dumpLog(eigCVal);
	MyUtility::dumpLog(eigCVec);
	MyUtility::dumpLog("\n");
#endif
#endif
	// compute tranform matrix
	Eigen::MatrixXf eigSqrtCVal = eigCVal.cwiseSqrt();
	Eigen::MatrixXf invEigSqrtSrcVal = invDiagMat(eigSqrtSrcVal);
	transformMat = eigSrcVec * invEigSqrtSrcVal * eigCVec * eigSqrtCVal * eigCVec.transpose() * invEigSqrtSrcVal * eigSrcVec.transpose();
#ifdef MY_LOG
#ifdef MY_DEBUG
	MyUtility::dumpLog("Dump T matrix ......\n");
	MyUtility::dumpLog(transformMat);
	MyUtility::dumpLogTime();
#endif
#endif
	return ErrorCode::RIGHT;
}

/**
* @brief make eigen value matrix positive
* @param Eigen::MatrixXf eigenValueMat: input diagonal eigen values matrix
* @return Eigen::MatrixXf: return revised positive eigen value matrix
*/
Eigen::MatrixXf MKLtransform::postiveEigenValues(Eigen::MatrixXf eigenValueMat) {
	for (int i = 0; i < eigenValueMat.rows(); i++) {
		if (eigenValueMat(i, i) <= 0) {
			eigenValueMat(i, i) = RELIGHTING_EPS;
		}
	}
	return eigenValueMat;
}

/**
* @brief inverse diagonal matrix
* @param Eigen::MatrixXf diagMat): input diagonal matrix
* @return Eigen::MatrixXf: return diagonal inverse matrix
*/
Eigen::MatrixXf MKLtransform::invDiagMat(Eigen::MatrixXf diagMat) {
	for (int i = 0; i < diagMat.rows(); i++) {
		diagMat(i, i) = 1 / diagMat(i, i);
	}
	return diagMat;
}

/**
* @brief estimate MKL transform
* @param int useMask: 1: used mask to discard black pixels 0: use all the pixel without mask (assume no black pixels)
* @return ColorTrans: return color tranform matrix
*/
ColorTrans MKLtransform::estimateMKLTransform(int useMask /* = 1 */) {
	ColorTrans trans;
	// transfer OpenCV to Eigen
	Eigen::MatrixXf srcMat;
	Eigen::MatrixXf tarMat;
	if (isSuperpixel == false) {
		if (useMask == 1) {
			cv::Mat mask;
			cv::bitwise_and(computeMask(this->src, 0, 720), computeMask(this->dst, 0, 720), mask);
			this->toEigenMat(this->dst, this->src, tarMat, srcMat, mask);
		}
		else {
			srcMat = toEigenMat(src);
			tarMat = toEigenMat(dst);
		}
	}
	else {
		srcMat = toEigenMat(spSrc);
		tarMat = toEigenMat(spDst);
	}
	Eigen::MatrixXf meanVecSrc, zeroMeanMatSrc, covMatSrc;
	Eigen::MatrixXf meanVecTar, zeroMeanMatTar, covMatTar;
	// compute meanVec, zeroMeanMat and covMat
	this->calcMeanCov(srcMat, meanVecSrc, zeroMeanMatSrc, covMatSrc);
	this->calcMeanCov(tarMat, meanVecTar, zeroMeanMatTar, covMatTar);
	// compute tranform matrix
	Eigen::MatrixXf transformR, transformT;
	calcMKLtransform(covMatSrc, covMatTar, transformR);
	transformT = -transformR * meanVecSrc.transpose() + meanVecTar.transpose();
	return ColorTrans(transformR, transformT);
}

/**
* @brief estimate MKL transform
* @param cv::Mat mask: input mask
* @return ColorTrans: return color tranform matrix
*/
ColorTrans MKLtransform::estimateMKLTransform(cv::Mat mask) {
	ColorTrans trans;
	// transfer OpenCV to Eigen
	Eigen::MatrixXf srcMat;
	Eigen::MatrixXf tarMat;
	if (isSuperpixel == false) {
		this->toEigenMat(this->dst, this->src, tarMat, srcMat, mask);
	}
	Eigen::MatrixXf meanVecSrc, zeroMeanMatSrc, covMatSrc;
	Eigen::MatrixXf meanVecTar, zeroMeanMatTar, covMatTar;
	// compute meanVec, zeroMeanMat and covMat
	this->calcMeanCov(srcMat, meanVecSrc, zeroMeanMatSrc, covMatSrc);
	this->calcMeanCov(tarMat, meanVecTar, zeroMeanMatTar, covMatTar);
	// compute tranform matrix
	Eigen::MatrixXf transformR, transformT;
	calcMKLtransform(covMatSrc, covMatTar, transformR);
	transformT = -transformR * meanVecSrc.transpose() + meanVecTar.transpose();
	return ColorTrans(transformR, transformT);
}
