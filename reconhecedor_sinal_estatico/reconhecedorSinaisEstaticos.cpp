
#include "SkeletonSensor.hpp"

// openCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
using namespace cv;

#include <iostream>
using namespace std;

// globals
SkeletonSensor* sensor;

const unsigned int XRES = 640;
const unsigned int YRES = 480;

const float DEPTH_SCALE_FACTOR = 255./4096.;

// defines the value about which thresholding occurs
const unsigned int BIN_THRESH_OFFSET = 10;

// defines the value about witch the region of interest is extracted
const unsigned int ROI_OFFSET = 70;

// median blur factor
const unsigned int MEDIAN_BLUR_K = 3;

// grasping threshold
const double GRASPING_THRESH = 0.9;

// colors
const Scalar COLOR_BLUE        = Scalar(240,40,0);
const Scalar COLOR_DARK_GREEN  = Scalar(0, 128, 0);
const Scalar COLOR_LIGHT_GREEN = Scalar(0,255,0);
const Scalar COLOR_YELLOW      = Scalar(0,128,200);
const Scalar COLOR_RED         = Scalar(0,0,255);

string depthFrame = "depthFrame";
string leftHandFrame = "leftHandFrame";
string rightHandFrame = "rightHandFrame";

// returns true if the hand is near the sensor area
bool handApproachingDisplayPerimeter(float x, float y)
{
    return (x > (XRES - ROI_OFFSET)) || (x < (ROI_OFFSET)) ||
           (y > (YRES - ROI_OFFSET)) || (y < (ROI_OFFSET));
}

// conversion from cvConvexityDefect
struct ConvexityDefect
{
    Point start;
    Point end;
    Point depth_point;
    float depth;
};

// Thanks to Jose Manuel Cabrera for part of this C++ wrapper function
/*void findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<ConvexityDefect>& convexDefects)
{
    if(hull.size() > 0 && contour.size() > 0)
    {
        CvSeq* contourPoints;
        CvSeq* defects;
        CvMemStorage* storage;
        CvMemStorage* strDefects;
        CvMemStorage* contourStr;
        CvConvexityDefect *defectArray = 0;

        strDefects = cvCreateMemStorage();
        defects = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq),sizeof(CvPoint), strDefects );

        //We transform our vector<Point> into a CvSeq* object of CvPoint.
        contourStr = cvCreateMemStorage();
        contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);
        for(int i = 0; i < (int)contour.size(); i++) {
            CvPoint cp = {contour[i].x,  contour[i].y};
            cvSeqPush(contourPoints, &cp);
        }

        //Now, we do the same thing with the hull index
        int count = (int) hull.size();
        //int hullK[count];
        int* hullK = (int*) malloc(count*sizeof(int));
        for(int i = 0; i < count; i++) { hullK[i] = hull.at(i); }
        CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);

        // calculate convexity defects
        storage = cvCreateMemStorage(0);
        defects = cvConvexityDefects(contourPoints, &hullMat, storage);
        defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*defects->total);
        cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
        //printf("DefectArray %i %i\n",defectArray->end->x, defectArray->end->y);

        //We store defects points in the convexDefects parameter.
        for(int i = 0; i<defects->total; i++){
            ConvexityDefect def;
            def.start       = Point(defectArray[i].start->x, defectArray[i].start->y);
            def.end         = Point(defectArray[i].end->x, defectArray[i].end->y);
            def.depth_point = Point(defectArray[i].depth_point->x, defectArray[i].depth_point->y);
            def.depth       = defectArray[i].depth;
            convexDefects.push_back(def);
        }

    // release memory
    cvReleaseMemStorage(&contourStr);
    cvReleaseMemStorage(&strDefects);
    cvReleaseMemStorage(&storage);

    }
}*/

static void colorizeDisparity( const Mat& gray, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f )
{
    CV_Assert( !gray.empty() );
    CV_Assert( gray.type() == CV_8UC1 );

    if( maxDisp <= 0 )
    {
        maxDisp = 0;
        minMaxLoc( gray, 0, &maxDisp );
    }

    rgb.create( gray.size(), CV_8UC3 );
    rgb = Scalar::all(0);
    if( maxDisp < 1 )
        return;

    for( int y = 0; y < gray.rows; y++ )
    {
        for( int x = 0; x < gray.cols; x++ )
        {
            uchar d = gray.at<uchar>(y,x);
            unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

            unsigned int hi = (H/60) % 6;
            float f = H/60.f - H/60;
            float p = V * (1 - S);
            float q = V * (1 - f * S);
            float t = V * (1 - (1 - f) * S);

            Point3f res;

            if( hi == 0 ) //R = V,  G = t,  B = p
                res = Point3f( p, t, V );
            if( hi == 1 ) // R = q, G = V,  B = p
                res = Point3f( p, V, q );
            if( hi == 2 ) // R = p, G = V,  B = t
                res = Point3f( t, V, p );
            if( hi == 3 ) // R = p, G = q,  B = V
                res = Point3f( V, q, p );
            if( hi == 4 ) // R = t, G = p,  B = V
                res = Point3f( V, p, t );
            if( hi == 5 ) // R = V, G = p,  B = q
                res = Point3f( q, p, V );

            uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
            uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
            uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

            rgb.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);
        }
    }
}

int main(int argc, char** argv)
{

    // initialize the kinect
    sensor = new SkeletonSensor();
    sensor->initialize();
    sensor->setPointModeToProjective();

    //Mat depthRaw(YRES, XRES, CV_16UC1);
    Mat depthShow(YRES, XRES, CV_8UC1);
    //Mat handDebug;
    
    // this vector holds the displayed images of the hands
    vector<Mat> debugFrames;

    // rectangle used to extract hand regions from depth map
    Rect roi;
    roi.width  = ROI_OFFSET*2;
    roi.height = ROI_OFFSET*2;

    namedWindow(depthFrame, CV_WINDOW_AUTOSIZE);
    namedWindow(leftHandFrame, CV_WINDOW_AUTOSIZE);
    namedWindow(rightHandFrame, CV_WINDOW_AUTOSIZE);


    int key = 0;
    while(key != 27 && key != 'q')
    {

        sensor->waitForDeviceUpdateOnUser();

        // update 16 bit depth matrix
        //memcpy(depthRaw.data, sensor->getDepthData(), XRES*YRES*2);
		VideoCapture capture( CV_CAP_OPENNI );
		capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ );
		capture.grab();
		capture.retrieve( depthShow, CV_CAP_OPENNI_DISPARITY_MAP );
		//depthRaw.convertTo(depthShow, CV_8UC1, DEPTH_SCALE_FACTOR);
        int handDepth = -1;

        for(int handI = 0; handI < 2; handI++)
        {

            if(sensor->getNumTrackedUsers() > 0)
            {
                Skeleton skel =  sensor->getSkeleton(sensor->getUID(0));
                SkeletonPoint hand;

                if( handI == 0)
                    hand = skel.leftHand;
                else
                    hand = skel.rightHand;
                if(hand.confidence == 1.0)
                {
                    handDepth = hand.z * (DEPTH_SCALE_FACTOR);
                    
					

                    if(!handApproachingDisplayPerimeter(hand.x, hand.y))
                    {
                        roi.x = hand.x - ROI_OFFSET;
                        roi.y = hand.y - ROI_OFFSET;
                    }
                }
            }
            else
                handDepth = -1;

            // extract hand from image
            Mat handMat(depthShow, roi);
            //Mat handMat = handCpy.clone();

            // binary threshold
            //if(handDepth != -1)
			    //handMat = (handMat > (handDepth - BIN_THRESH_OFFSET)) & (handMat < (handDepth + BIN_THRESH_OFFSET));

            // last pre-filtering step, apply median blur
            //medianBlur(handMat, handMat, MEDIAN_BLUR_K);

            //imshow( "colorized disparity map", validColorDisparityMap );


            // create debug image of thresholded hand and cvt to RGB so hints show in color
            //handDebug = handMat.clone();
            
            Mat colorDisparityMap;
            colorizeDisparity(handMat, colorDisparityMap, -1 );
            Mat validColorDisparityMap;
            colorDisparityMap.copyTo( validColorDisparityMap, handMat != 0 );
			cv::Size a(1,1);
			GaussianBlur(handMat, handMat, a, 1); 

			/*capture.retrieve( disparityMap, CV_CAP_OPENNI_DISPARITY_MAP );
            Mat colorDisparityMap;
            colorizeDisparity( disparityMap, colorDisparityMap, -1 );
            Mat validColorDisparityMap;
            colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );
            imshow( "colorized disparity map", validColorDisparityMap );*/

			
			debugFrames.push_back(validColorDisparityMap);
            //cvtColor(debugFrames[handI], debugFrames[handI], CV_GRAY2RGB);

			Mat handCpy = handMat.clone();
            std::vector< std::vector<Point> > contours;
            //findContours(handCpy, contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
			findContours(handCpy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            if (contours.size()) {
                for (int i = 0; i < contours.size(); i++) {
                    vector<Point> contour = contours[i];
                    Mat contourMat = Mat(contour);
                    double cArea = contourArea(contourMat);

                    if(cArea > 3000) // likely the hand
                    {
                        Scalar center = mean(contourMat);
                        //Point centerPoint = Point(center.val[0], center.val[1]);

                        // approximate the contour by a simple curve
                        vector<Point> approxCurve;
                        approxPolyDP(contourMat, approxCurve, 10, true);

                        //vector< vector<Point> > debugContourV;
                        //debugContourV.push_back(approxCurve);
                        //drawContours(debugFrames[handI], debugContourV, 0, COLOR_DARK_GREEN, 2);

                        vector<int> hull;
                        convexHull(Mat(approxCurve), hull, false, false);

                        // draw the hull points
                        //for(int j = 0; j < hull.size(); j++)
                        //{
                        //   int index = hull[j];
                        //    circle(debugFrames[handI], approxCurve[index], 3, COLOR_YELLOW, 2);
                        //}

                        // find convexity defects
                        //vector<ConvexityDefect> convexDefects;
                        //findConvexityDefects(approxCurve, hull, convexDefects);
                        //printf("Number of defects: %d.\n", (int) convexDefects.size());

                        //for(int j = 0; j < convexDefects.size(); j++)
                        //{
                            //circle(debugFrames[handI], convexDefects[j].depth_point, 3, COLOR_BLUE, 2);

                        //}
                        
                        // assemble point set of convex hull
                        vector<Point> hullPoints;
                        for(int k = 0; k < hull.size(); k++)
                        {
                            int curveIndex = hull[k];
                            Point p = approxCurve[curveIndex];
                            hullPoints.push_back(p);
                        }

                        // area of hull and curve
                        double hullArea  = contourArea(Mat(hullPoints));
                        double curveArea = contourArea(Mat(approxCurve));
                        double handRatio = curveArea/hullArea;
						//printf("\n %d", handRatio);
                        // hand is grasping
                        if(handRatio <= GRASPING_THRESH)
                            //circle(debugFrames[handI], centerPoint, 5, COLOR_LIGHT_GREEN, 5);
							printf("\nOOOOOOOOOOOOOOOO");
                            //circle(debugFrames[handI], centerPoint, 5, COLOR_RED, 5);
                    }
                } // contour conditional
            } // hands loop
        }

        imshow(depthFrame, depthShow);
        
        if(debugFrames.size() >= 2 )
        {
            resize(debugFrames[0], debugFrames[0], Size(), 3, 3);
            resize(debugFrames[1], debugFrames[1], Size(), 3, 3);
            imshow(leftHandFrame,  debugFrames[0]);
            imshow(rightHandFrame,  debugFrames[1]);
            debugFrames.clear();
        }


        key = waitKey(10);

    }

    delete sensor;

    return 0;
}

/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

int main()
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    Mat labelsMat(4, 1, CV_32FC1, labels);

    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

    Vec3b green(0,255,0), blue (255,0,0);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                 image.at<Vec3b>(i,j)  = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);

}*/
