#include "RelightingInterpolation.h"
#include <opencv2/ximgproc.hpp>

std::string edgeModelname = "/home/ubuntu/ShaneYuan/project/mantis_api_code/mantis_camera_color_correction/model/model.yml";

// edge detector class
class EdgeDetector {
public:
    static cv::Mat structuredEdgeDetector(cv::Mat img, std::string modelname) {
        cv::Mat edges;
        cv::Ptr<cv::ximgproc::StructuredEdgeDetection> ptr = cv::ximgproc::createStructuredEdgeDetection(modelname);
        img.convertTo(img, cv::DataType<float>::type, 1/255.0);
        ptr->detectEdges(img, edges);
        edges = edges * 255;
        edges.convertTo(edges, CV_8U);
        return edges;
    }

    static cv::Mat laplacianEdgeDetector(cv::Mat img) {
        cv::Mat edges, imgGray;
        cv::cvtColor(img, imgGray, CV_BGR2GRAY);
        cv::Laplacian(imgGray, edges, CV_8U, 5);
        cv::convertScaleAbs(edges, edges);
        return edges;
    }
};

int localviewCamLocalization(cv::Mat refImg, cv::Mat localImg, float scale, cv::Mat& refBlk, cv::Rect& rect) {
    // compute new size
    cv::Mat localImg2;
    cv::Size smallsize = cv::Size(static_cast<float>(localImg.cols) * scale, static_cast<float>(localImg.rows) * scale);
    cv::resize(localImg, localImg2, smallsize);

    cv::Mat local = localImg2.clone();
    cv::Mat localEdge = EdgeDetector::structuredEdgeDetector(local, edgeModelname);

    cv::Mat ref = refImg.clone();
    cv::Mat refEdge = EdgeDetector::structuredEdgeDetector(ref, edgeModelname);

    // zncc matching
    cv::Mat resultRGB, resultEdge, result;
    cv::matchTemplate(ref, local, resultRGB, cv::TM_CCOEFF_NORMED, cv::Mat());
    cv::matchTemplate(refEdge, localEdge, resultEdge, cv::TM_CCOEFF_NORMED, cv::Mat());
    result = resultRGB.mul(resultEdge);

    double minVal, maxVal;
    cv::Point2i minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
    rect = cv::Rect(maxLoc.x, maxLoc.y, smallsize.width, smallsize.height);
    refBlk = refImg(rect).clone();

    return 0;
}

int main(int argc, char* argv[]) {
    std::string inputdir = "/home/ubuntu/ShaneYuan/project/mantis_api_code/mantis_camera_color_correction/data/";
    std::string outdir = "/home/ubuntu/ShaneYuan/project/mantis_api_code/mantis_camera_color_correction/result/";
    std::vector<std::string> localnames;
    for (int i = 0; i < 18; i ++) {
        localnames.push_back(cv::format("%s/%02d_localview.jpg", inputdir.c_str(), i));
    }
    std::string refname = cv::format("%s/00_refimg.jpg", inputdir.c_str());
    float scale = 0.14;

    // localview localization
    cv::Mat refBlk;
    cv::Mat refImg = cv::imread(refname);

    for (int i = 0; i < localnames.size(); i ++) {
        std::cout << "Process image " << i << "..." << std::endl;
        cv::Mat localImg = cv::imread(localnames[i]);
        cv::Rect rect;
        localviewCamLocalization(refImg, localImg, scale, refBlk, rect);
        // color correction
        MKLtransform mklTransform;
        mklTransform.setInput(refImg, refBlk);
        ColorTrans colorTrans = mklTransform.estimateMKLTransform();
        cv::Mat result = colorTrans.apply(localImg);
        // write results to file
        cv::imwrite(cv::format("%s/%02d_localview.jpg", outdir.c_str(), i), localImg);
        cv::imwrite(cv::format("%s/%02d_refblk.jpg", outdir.c_str(), i),refBlk);
        cv::imwrite(cv::format("%s/%02d_relight.jpg", outdir.c_str(), i), result);
    }

    return 0;
}