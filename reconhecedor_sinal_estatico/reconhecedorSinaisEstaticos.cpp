#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include "SkeletonSensor.hpp"
#include <string>
#include <iostream>

using namespace std;
using namespace cv;
//define a resolucao do kinect
const unsigned int XRES = 640;
const unsigned int YRES = 480;
//define o valor para extração da ROI (Region of Interest)
const unsigned int ROI_OFFSET = 70;
//define o valor para extração da coloração produzida pela disparidade
const unsigned int COLOR_OFFSET = 170;
//define a quantidade de repetições para coletas
const unsigned int REPETICAO = 2;
//define o sensor para captura do esqueleto via OpenNI/PrimeSense
SkeletonSensor* sensor;
//define o classificador SVM (Support Vector Machines)
CvSVM svm;
CvSVMParams params;
//define a referência para as janelas de saída
string frameProfundidade = "frameProfundidade";
string frameMaoEsquerda = "frameMaoEsquerda";
string frameMaoDireita = "frameMaoDireita";

//coloriza a disparidade obtida com câmera de profundidade
//função retirada de exemplo do OpenCV
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
			
            uchar b = (uchar)((std::max)(0.f, (std::min)(res.x, 1.f)) * 255.f);
            uchar g = (uchar)((std::max)(0.f, (std::min)(res.y, 1.f)) * 255.f);
            uchar r = (uchar)((std::max)(0.f, (std::min)(res.z, 1.f)) * 255.f);

            rgb.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);
        }
    }
}

//retorna true se a mão estiver se aproximando do perímetro do frame
bool maoProximaPerimetro(float x, float y)
{
    return (x > (XRES - ROI_OFFSET)) || (x < (ROI_OFFSET)) 
		|| (y > (YRES - ROI_OFFSET)) || (y < (ROI_OFFSET));
}//bool maoProximaPerimetro(float x, float y)

int main(int argc, char** argv)
{
    //inicialização do kinect pelo driver e do sensor de esqueleto
    sensor = new SkeletonSensor();
    sensor->initialize();
    sensor->setPointModeToProjective();

	//objeto Mat que recebe os dados de profundidade do kinect
    Mat depthShow(YRES, XRES, CV_8UC1);

	//DESABILITAR O CÓDIGO ABAIXO QUANDO PARA NÃO PREJUDICAR O EXPERIMENTO
	//área dos frames das mãos
	int area = 140*140;
	Mat dadosTreinamento(REPETICAO, area, CV_32FC1);
	float label[REPETICAO];
	int quantidadeTreinamento = 0;

    //retângulo para extração da região das mãos
    Rect roi;
    roi.width  = ROI_OFFSET*2;
    roi.height = ROI_OFFSET*2;

    //vetor com a imagem das duas mãos
    vector<Mat> debugFrames;

	//cria os frames que serão usados
    namedWindow(frameProfundidade, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoEsquerda, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoDireita, CV_WINDOW_AUTOSIZE);

	//interação com o teclado
	int teclado = 0;
    
	while(1) {
        sensor->waitForDeviceUpdateOnUser();

		//captura os dados do kinect para uso com o OpenCV
		VideoCapture capture(CV_CAP_OPENNI);
		capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
		capture.grab();
		capture.retrieve(depthShow, CV_CAP_OPENNI_DISPARITY_MAP);

        for (int indiceMao = 0; indiceMao < 2; indiceMao++) {
            if (sensor->getNumTrackedUsers() > 0) {
				//obtém o esqueleto
                Skeleton esqueleto =  sensor->getSkeleton(sensor->getUID(0));
                SkeletonPoint mao;
				//define a mão com base no índice
                if (indiceMao == 0) {
                    mao = esqueleto.leftHand;
				} else {
					mao = esqueleto.rightHand;
				}//if (indiceMao == 0)
                if (mao.confidence == 1.0) {
					//ajusta a região de interesse (mão) pela movimentação
                    if (!maoProximaPerimetro(mao.x, mao.y)) {
                        roi.x = mao.x - ROI_OFFSET;
                        roi.y = mao.y - ROI_OFFSET;
                    }//if (!maoProximaPerimetro(mao.x, mao.y))
                }//if (mao.confidence == 1.0)

				Mat colorDisparityMap;
				Mat validColorDisparityMap;
				Mat handMat(depthShow, roi);

				//coloriza a disparidade encontrada pela câmera de profundidade
				colorizeDisparity(handMat, colorDisparityMap, -1);
				colorDisparityMap.copyTo(validColorDisparityMap, handMat != 0);
				
				//aplica um blur pra tentar conter o ruído
				cv::Size a(1,1);
				GaussianBlur(validColorDisparityMap, validColorDisparityMap, a, 1); 
				//medianBlur(handMat, handMat, MEDIAN_BLUR_K);

				int i,j;
				for (i = 0; i < validColorDisparityMap.rows; i++) {
					for ( j = 0; j < validColorDisparityMap.cols; j++) {
						if(validColorDisparityMap.at<Vec3b>(i, j)[2] 
						< (validColorDisparityMap.at<Vec3b>(i, j)[0])+COLOR_OFFSET
						|| validColorDisparityMap.at<Vec3b>(i, j)[2] 
						< (validColorDisparityMap.at<Vec3b>(i, j)[1])+COLOR_OFFSET) {
							validColorDisparityMap.at<Vec3b>(i, j)[0] = 0;
							validColorDisparityMap.at<Vec3b>(i, j)[1] = 0;
							validColorDisparityMap.at<Vec3b>(i, j)[2] = 0;

						}//if(validColorDisparityMap.at<Vec3b>(i, j)[2] ...
					}//for ( j = 0; j < validColorDisparityMap.cols; j++)
				}//for( i = 0; i < validColorDisparityMap.rows; i++)

				debugFrames.push_back(validColorDisparityMap);
			}//if (sensor->getNumTrackedUsers() > 0)
        }//for (int indiceMao = 0; indiceMao < 2; indiceMao++)

		teclado = waitKey(10);

		if (teclado == 'q' || teclado == 'w') {
			int i,j,k=0;
			for(i = 0; i < debugFrames[0].rows; i++) {
				for (j = 0; j < debugFrames[0].cols; j++) {
					float G = debugFrames[0].at<Vec3b>(i, j)[1];
					float R = debugFrames[0].at<Vec3b>(i, j)[2];
					dadosTreinamento.at<float>(quantidadeTreinamento,k++) = (R+G);
				}//for (j = 0; j < debugFrames[0].cols; j++)
			}//for(i = 0; i < debugFrames[0].rows; i++)

			if (teclado == 'q') {
				printf("\nFrame capturado para treinamento correto!");
				label[quantidadeTreinamento] = 1.0;
			} else {
				printf("\nFrame capturado para treinamento errado!");
				label[quantidadeTreinamento] = -1.0;
			}
			quantidadeTreinamento++;
		} else if (teclado == 'z') {
			break;
		}//if (teclado == 27 || teclado == 'q')

        imshow(frameProfundidade, depthShow);
        
        if (debugFrames.size() >= 2) {
            resize(debugFrames[0], debugFrames[0], Size(), 3, 3);
            resize(debugFrames[1], debugFrames[1], Size(), 3, 3);
			imshow(frameMaoDireita,  debugFrames[0]);
            imshow(frameMaoEsquerda,  debugFrames[1]);
            debugFrames.clear();
        }//if (debugFrames.size() >= 2)


    }

	Mat labelTreinamento(REPETICAO, 1, CV_32FC1, label);

	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	svm.train(dadosTreinamento, labelTreinamento, Mat(), Mat(), params);

	svm.save("treinamento_linear.xml");

	//printf("\n salvando treinamento");
	// Declare what you need
	//cv::FileStorage file("treinamento1_esq.xml", cv::FileStorage::WRITE);
	//printf("\n salvando treinamento2");
	// Write to file!
	//file << "MaoEsquerda" << debugFrames[0];

	//file.release();

	debugFrames.clear();

    delete sensor;

    return 0;
}


/*#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main()
{
    Mat rawTreinamento;
	cv::FileStorage file("treinamento1_esq.xml", FileStorage::READ);
	file["MaoEsquerda"] >> rawTreinamento;

	if (!file.isOpened()) {
		printf("\n aeheauheua");
	}

	int area = 140*140;
	Mat dadosTreinamento(2, area, CV_32FC1);

	int i,j,k=0;
	for(i = 0; i < rawTreinamento.rows; i++) {
		for (j = 0; j < rawTreinamento.cols; j++) {
			float B = 0;
			float G = rawTreinamento.at<Vec3b>(i, j)[1];
			float R = rawTreinamento.at<Vec3b>(i, j)[2];
			dadosTreinamento.at<float>(1,k) = B;
			dadosTreinamento.at<float>(0,k++) = (R+G);
		}
	}

	float label[2] = {1.0, -1.0};
	Mat labelTreinamento(2, 1, CV_32FC1, label); 
//system("PAUSE");
    
	CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	//printf("%d %d", dadosTreinamento.size(), labelTreinamento.size());
//	system("PAUSE");
    CvSVM SVM;
    SVM.train(dadosTreinamento, labelTreinamento, Mat(), Mat(), params);
	
	Mat teste(1, area, CV_32FC1);
	Mat teste2(1, area, CV_32FC1);
	k = 0;
	for(i = 0; i < rawTreinamento.rows; i++) {
		for (j = 0; j < rawTreinamento.cols; j++) {
			float B = 0;
			float G = rawTreinamento.at<Vec3b>(i, j)[1];
			float R = rawTreinamento.at<Vec3b>(i, j)[2];
			teste2.at<float>(0,k) = B;
			teste.at<float>(0,k++) = (R+G);
		}
	}

	float response = SVM.predict(teste);
	cout << response << endl;
	response = SVM.predict(teste2);
	cout << response << endl;
	system("PAUSE");
	/*
    // Set up training data
    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    Mat labelsMat(4, 1, CV_32FC1, labels);
	//float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
    //Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

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



				/*Mat handCpy = handMat.clone();
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
							//drawContours(debugFrames[indiceMao], debugContourV, 0, COLOR_DARK_GREEN, 2);

							vector<int> hull;
							convexHull(Mat(approxCurve), hull, false, false);

							// draw the hull points
							//for(int j = 0; j < hull.size(); j++)
							//{
							//   int index = hull[j];
							//    circle(debugFrames[indiceMao], approxCurve[index], 3, COLOR_YELLOW, 2);
							//}

							// find convexity defects
							vector<ConvexityDefect> convexDefects;
							findConvexityDefects(approxCurve, hull, convexDefects);
							//printf("Number of defects: %d.\n", (int) convexDefects.size());

							//for(int j = 0; j < convexDefects.size(); j++)
							//{
								//circle(debugFrames[indiceMao], convexDefects[j].depth_point, 3, COLOR_BLUE, 2);

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
							if(handRatio <= GRASPING_THRESH && convexDefects.size() > 0)
								//circle(debugFrames[indiceMao], centerPoint, 5, COLOR_LIGHT_GREEN, 5);
								printf("\nOOOOOOOOOOOOOOOO %d | %d", convexDefects.size(), hull.size());
								//circle(debugFrames[indiceMao], centerPoint, 5, COLOR_RED, 5);
						}
					} // contour conditional
				} // hands loop*/