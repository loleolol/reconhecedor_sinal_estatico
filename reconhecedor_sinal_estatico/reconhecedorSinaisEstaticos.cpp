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
const unsigned int REPETICAO = 30;
//define o índice das duas mãos
const unsigned int MAO_ESQUERDA = 1;
const unsigned int MAO_DIREITA = 0;
//define os índices das escalas BGR
const unsigned int B = 0;
const unsigned int G = 1;
const unsigned int R = 2;
//define os sinais reconhecidos
const float SINAL_A = 1.0;
const float SINAL_B = 2.0;
const float SINAL_NADA = -1.0;
//define a quantidade de repetições para delay da predição SVM
const unsigned int DELAY = 10;

const Scalar COLOR_LIGHT_GREEN = Scalar(0,255,0);
//define o sensor para captura do esqueleto via OpenNI/PrimeSense
SkeletonSensor* sensor;
//define o classificador SVM (Support Vector Machines)
CvSVM svm;
CvSVMParams params;
//define a referência para as janelas de saída
string frameImagem = "Imagem";
string frameProfundidade = "Profundidade";
string frameMaoEsquerda = "MaoEsquerda";
string frameMaoDireita = "MaoDireita";
string window = "window";

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
	//define o modo de operação da aplicação
	int TREINAMENTO = 1;

    //inicialização do kinect pelo driver e do sensor de esqueleto
    sensor = new SkeletonSensor();
    sensor->initialize();
    sensor->setPointModeToProjective();

	//objeto Mat que recebe os dados de profundidade do kinect
    Mat mapaProfundidade(YRES, XRES, CV_8UC1);
	Mat imagemBGR;
	//área dos frames das mãos
	int area = 140*140;
	Mat dadosTreinamento(REPETICAO, area, CV_32FC1);
	float label[REPETICAO];
	int quantidadeTreinamento = 0;
	//cor da borda
	Scalar color = Scalar(255,255,255);

    //retângulo para extração da região das mãos
    Rect roi;
    roi.width  = ROI_OFFSET*2;
    roi.height = ROI_OFFSET*2;

    //vetor com a imagem das duas mãos
    vector<Mat> debugFrames;

	//cria os frames que serão usados
	namedWindow(frameImagem, CV_WINDOW_AUTOSIZE);
    namedWindow(frameProfundidade, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoEsquerda, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoDireita, CV_WINDOW_AUTOSIZE);
	//interação com o teclado
	int teclado = 0;
	int delay = -1;

	//carrega o treinamento salvo pelo SVM para classificar os sinais
	printf("\nTreinamento existente carregado!\n");
	svm.load("treinamento_linear.xml");

	while (1) {
        sensor->waitForDeviceUpdateOnUser();

		//captura os dados do kinect para uso com o OpenCV
		VideoCapture capture(CV_CAP_OPENNI);
		capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
		capture.grab();
		capture.retrieve(imagemBGR, CV_CAP_OPENNI_BGR_IMAGE);
		capture.retrieve(mapaProfundidade, CV_CAP_OPENNI_DISPARITY_MAP);
		delay++;
		if (delay > DELAY) {
			delay = 0;
		}

        for (int indiceMao = 0; indiceMao < 2; indiceMao++) {
            if (sensor->getNumTrackedUsers() > 0) {
				//obtém o esqueleto
                Skeleton esqueleto =  sensor->getSkeleton(sensor->getUID(0));
                SkeletonPoint mao;
				//define a mão com base no índice
                if (indiceMao == MAO_DIREITA) {
                    mao = esqueleto.leftHand;//direita para o usuário
				} else {
					mao = esqueleto.rightHand;//esquerda para o usuário
				}//if (indiceMao == MAO_DIREITA)
                if (mao.confidence == 1.0) {
					//ajusta a região de interesse (mão) pela movimentação
                    if (!maoProximaPerimetro(mao.x, mao.y)) {
                        roi.x = mao.x - ROI_OFFSET;
                        roi.y = mao.y - ROI_OFFSET;
                    }//if (!maoProximaPerimetro(mao.x, mao.y))
                }//if (mao.confidence == 1.0)
				Mat mapaDisparidadeColorido;
				Mat mapaDisparidadeColoridoValido;
				Mat mapaMao(mapaProfundidade, roi);

				//coloriza a disparidade encontrada pela câmera de profundidade
				colorizeDisparity(mapaMao, mapaDisparidadeColorido, -1);
				mapaDisparidadeColorido.copyTo(mapaDisparidadeColoridoValido, mapaMao != 0);
				
				//applyColorMap(mapaMao, mapaDisparidadeColoridoValido, COLORMAP_BONE);
				//aplica um blur pra tentar conter o ruído
				//cv::Size a(5,5);
				//GaussianBlur(mapaDisparidadeColoridoValido, mapaDisparidadeColoridoValido, a, 1); 
				//blur(mapaDisparidadeColoridoValido, mapaDisparidadeColoridoValido, a);
				medianBlur(mapaDisparidadeColoridoValido, mapaDisparidadeColoridoValido, 5);
				int i,j;
				for (i = 0; i < mapaDisparidadeColoridoValido.rows; i++) {
					for (j = 0; j < mapaDisparidadeColoridoValido.cols; j++) {
						if(mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[R] 
						< (mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[B])+COLOR_OFFSET
						|| mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[R] 
						< (mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[G])+COLOR_OFFSET) {
							mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[B] = 0;
							mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[G] = 0;
							mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[R] = 0;
						}//if(mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[R] ...
					}//for ( j = 0; j < mapaDisparidadeColoridoValido.cols; j++)
				}//for( i = 0; i < mapaDisparidadeColoridoValido.rows; i++)
							
				Mat mapaCannyEdge;
				cvtColor(mapaDisparidadeColoridoValido, mapaCannyEdge, CV_BGR2GRAY);
				threshold(mapaCannyEdge, mapaCannyEdge, 50, 255, THRESH_BINARY+THRESH_OTSU);
				Canny(mapaCannyEdge, mapaCannyEdge, 100, 100*2);

				std::vector<std::vector<Point>> contornos;
				findContours(mapaCannyEdge, contornos, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
				//resize(canny_output, canny_output, Size(), 3, 3);
				

				if (contornos.size()) {
					int indiceMaior = 0;
					double maiorArea = 0;
					double area;

					for (int i = 0; i < contornos.size(); i++) {
						drawContours(mapaDisparidadeColoridoValido, contornos, i, color, 2, 8, noArray(), 0, Point() );

						area = contourArea(contornos[i]);

						if (maiorArea < area) {
							indiceMaior = i;
							maiorArea = area;
						}//if (maiorArea > cArea)
					}//for (int i = 0; i < contornos.size(); i++)

					
					vector<Point> contorno = contornos[indiceMaior];
					area = contourArea(contorno);

					Scalar centro = mean(Mat(contorno));
					Point pontoCentral = Point(centro.val[0], centro.val[1]);
					circle(mapaDisparidadeColoridoValido, pontoCentral, 2, COLOR_LIGHT_GREEN, 2);

					//vector<Point> curva;
					//approxPolyDP(mapaContorno, curva, 10, true);
					vector<vector<Point>> corpo(contornos.size());
					vector<vector<int>> corpoI(contornos.size());
					convexHull(contorno, corpo[0], false);
					convexHull(contorno, corpoI[0], false);
					
					drawContours(mapaDisparidadeColoridoValido, corpo, 0, COLOR_LIGHT_GREEN, 1, 8, vector<Vec4i>(), 0, Point());

					vector<vector<Vec4i>> defeitosConvexos(contornos.size());
					convexityDefects(Mat(contornos[indiceMaior]), corpoI[0], defeitosConvexos[0]);
					
					for(int j = 0; j < defeitosConvexos[0].size(); j++) {
						int k = defeitosConvexos[0][j].val[2];
						Point p(contorno[k]);
						circle(mapaDisparidadeColoridoValido, p, 3, COLOR_LIGHT_GREEN, 2);

					}

				}

				//Mat handMat;
				//colorizeDisparity(mapaMao, handMat, -1);
				//Mat handMat = mapaMao.clone();
				//findContours(mapaDisparidadeColoridoValido, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
				
				//clona o mapa para poder inverter, no reconhecimento da mão esquerda
				Mat mapaTemporario = mapaDisparidadeColoridoValido.clone();

				if (indiceMao == MAO_ESQUERDA) {
					flip(mapaTemporario, mapaTemporario, 1);
				}//if (indiceMao == MAO_ESQUERDA)

				Mat dadosTeste(1, area, CV_32FC1);
				int k=0;
				for (i = 0; i < mapaTemporario.rows; i++) {
					for ( j = 0; j < mapaTemporario.cols; j++) {
						float tG = mapaTemporario.at<Vec3b>(i, j)[G];
						float tR = mapaTemporario.at<Vec3b>(i, j)[R];
						dadosTeste.at<float>(0,k++) = (tR+tG);
					}//for ( j = 0; j < mapaTemporario.cols; j++)
				}//for( i = 0; i < mapaTemporario.rows; i++)
			
				//realiza a classificação após delay
				if (TREINAMENTO == 0 && delay == 0) {
					if (indiceMao == MAO_ESQUERDA) {
						printf("\nMao esquerda: ");
					} else {
						printf("\nMao direita: ");
					}//if (indiceMao == 1)
					float resposta = svm.predict(dadosTeste);
					if (resposta == SINAL_B) {
						printf("LETRA B");
					} else if (resposta == SINAL_A) {
						printf("LETRA A");
					} else {
						printf("NADA");
					}//if (resposta == SINAL_B)
				}//if (delay == 0)

				debugFrames.push_back(mapaDisparidadeColoridoValido);
			}//if (sensor->getNumTrackedUsers() > 0)
        }//for (int indiceMao = 0; indiceMao < 2; indiceMao++)

		teclado = waitKey(10);

		if (teclado == 'q' || teclado == 'w' || teclado == 'e') {
			int i,j,k=0;
			for(i = 0; i < debugFrames[0].rows; i++) {
				for (j = 0; j < debugFrames[0].cols; j++) {
					float G = debugFrames[0].at<Vec3b>(i, j)[1];
					float R = debugFrames[0].at<Vec3b>(i, j)[2];
					dadosTreinamento.at<float>(quantidadeTreinamento,k++) = (R+G);
				}//for (j = 0; j < debugFrames[0].cols; j++)
			}//for(i = 0; i < debugFrames[0].rows; i++)

			if (teclado == 'q') {
				printf("\nFrame capturado para treinamento da letra A!");
				label[quantidadeTreinamento] = SINAL_A;
			} else if (teclado == 'w') {
				printf("\nFrame capturado para treinamento da letra B!");
				label[quantidadeTreinamento] = SINAL_B;
			} else {
				printf("\nFrame capturado para treinamento de casos errados!");
				label[quantidadeTreinamento] = SINAL_NADA;
			}
			quantidadeTreinamento++;
		} else if (teclado == 'x') {
			if (TREINAMENTO == 0) {
				printf("\nIniciando novo treinamento!");
				TREINAMENTO = 1;
			} else {
				printf("\nIniciando reconhecimento!");
				TREINAMENTO = 0;
			}//if (TREINAMENTO == 0)
		} else if (teclado == 'z') {
			break;
		}//if (teclado == 27 || teclado == 'q')

		imshow(frameImagem, imagemBGR);
        imshow(frameProfundidade, mapaProfundidade);
        
        if (debugFrames.size() >= 2) {
            resize(debugFrames[0], debugFrames[0], Size(), 3, 3);
            resize(debugFrames[1], debugFrames[1], Size(), 3, 3);
			imshow(frameMaoDireita,  debugFrames[0]);
            imshow(frameMaoEsquerda,  debugFrames[1]);
            debugFrames.clear();
        }//if (debugFrames.size() >= 2)
    }//while (1)

	if (TREINAMENTO == 1) {
		Mat labelTreinamento(REPETICAO, 1, CV_32FC1, label);

		params.svm_type    = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

		svm.train(dadosTreinamento, labelTreinamento, Mat(), Mat(), params);

		printf("\nGravando treinamento!");
		svm.save("treinamento_linear.xml");
	}//if (TREINAMENTO == 1)
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