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
const unsigned int COR_OFFSET = 170;
//define o valor para equalizar números em 3 casas d
const unsigned int NUMERO_OFFSET = 100;
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
//define a quantidade de repetições para coletas
const unsigned int REPETICAO = 30;
//define o kernel a ser utilizado
//const int SVM_KERNEL = CvSVM::LINEAR;
//const int SVM_KERNEL = CvSVM::POLY;
//const int SVM_KERNEL = CvSVM::RBF;
//define cores
const Scalar COR_VERDE    = Scalar(0,255,0);
const Scalar COR_AZUL     = Scalar(240,40,0);
const Scalar COR_AMARELO  = Scalar(0,128,200);
const Scalar COR_VERMELHO = Scalar(0,0,255);
const Scalar COR_BRANCO   = Scalar(255,255,255);

//coloriza a disparidade obtida com câmera de profundidade - função retirada de exemplo do OpenCV
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

// Concatena dois números
unsigned concatenarNumeros(unsigned a, unsigned b) 
{
    unsigned c = 10;
    while (b >= c) {
        c *= 10;
	}//while (b >= c)
    return a * c + b;        
}//unsigned concatenarNumeros(unsigned a, unsigned b)

//imprime a resposta conforme resultado
void imprimirResposta(float resposta)
{
	if (resposta == SINAL_B) {
		printf("LETRA B");
	} else if (resposta == SINAL_A) {
		printf("LETRA A");
	} else {
		printf("NADA");
	}//if (resposta == SINAL_B)
}//void imprimirResposta(float resposta)

//busca o nome do arquivo para carregar/salvar
char *buscarNomeArquivo(int kernel)
{
	if (kernel == CvSVM::LINEAR) {
		return "treinamento_1vs1_linear.xml";
	} else if (kernel == CvSVM::POLY) {
		return "treinamento_1vs1_polinomial.xml";
	} else if (kernel == CvSVM::RBF) {
		return "treinamento_1vs1_radial.xml";
	}//if (SVM_KERNEL == CvSVM::LINEAR)
}//string buscarNomeArquivo(int kernel)

int main(int argc, char** argv)
{
	//define o classificador SVM (Support Vector Machines) do OpenCV (1-vs-1)
	CvSVM linearSVM;
	CvSVM polinomialSVM;
	CvSVM radialSVM;
	CvSVMParams SVMparams;
	//define a referência para as janelas de saída
	string frameProfundidade = "Profundidade com Esqueleto";
	string frameMaoEsquerda = "Mao Esquerda";
	string frameMaoDireita = "Mao Direita";
	string frameMaoEsquerdaBGR = "Referencia E BGR";
	string frameMaoDireitaBGR = "Referencia D BGR";

	//define o modo de operação padrão
	int TREINAMENTO = 1;

	//define o sensor para captura do esqueleto via OpenNI/PrimeSense
    //inicialização do kinect pelo driver e do sensor de esqueleto
    SkeletonSensor* sensor = new SkeletonSensor();
    sensor->initialize();
    sensor->setPointModeToProjective();
	
	//mapa de profundidade que receberá os dados do kinect
    Mat mapaProfundidade(YRES, XRES, CV_8UC1);
	//mapa de profundidade sem cor que aceita o desenho do esqueleto colorido
	Mat mapaProfundidadeEsqueleto(YRES, XRES, CV_32FC1);
	//mapa da imagem BGR
	Mat mapaBGR;
    //retângulo para extração da região das mãos
    Rect roi;
    roi.width  = ROI_OFFSET*2;
    roi.height = ROI_OFFSET*2;

    //vetores com a imagem das duas mãos
    vector<Mat> mapaMaos, mapaMaosBGR;

	//área dos frames das mãos TODO
	int areaMapa = 140*140;
	Mat dadosTreinamento(REPETICAO*2, areaMapa, CV_32FC1);
	float label[REPETICAO*2];
	int quantidadeTreinamentoA = 0;
	int quantidadeTreinamentoB = 0;
	int quantidadeTreinamento = 0;

	//cria os frames que serão usados
    namedWindow(frameProfundidade, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoEsquerda, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoDireita, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoEsquerdaBGR, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoDireitaBGR, CV_WINDOW_AUTOSIZE);

	//interação com o teclado
	int teclado = 0;
	
	//carrega o treinamento salvo pelo SVM para classificar os sinais
	linearSVM.load(buscarNomeArquivo(CvSVM::LINEAR));
	polinomialSVM.load(buscarNomeArquivo(CvSVM::POLY));
	radialSVM.load(buscarNomeArquivo(CvSVM::RBF));

	printf("\nTreinamento existente carregado!\n");

	while (1) {
        sensor->waitForDeviceUpdateOnUser();

		//captura os dados do kinect para uso com o OpenCV
		VideoCapture capture(CV_CAP_OPENNI);
		capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
		capture.grab();
		capture.retrieve(mapaBGR, CV_CAP_OPENNI_BGR_IMAGE);
		capture.retrieve(mapaProfundidade, CV_CAP_OPENNI_DISPARITY_MAP);

        for (int indiceMao = 0; indiceMao < 2; indiceMao++) {
            if (sensor->getNumTrackedUsers() > 0) {
				//obtém o esqueleto
                Skeleton esqueleto =  sensor->getSkeleton(sensor->getUID(0));
                SkeletonPoint mao;
				//define a mão com base no índice
                if (indiceMao == MAO_DIREITA) {
					//direita para o usuário
                    mao = esqueleto.leftHand;
				} else {
					//esquerda para o usuário
					mao = esqueleto.rightHand;
				}//if (indiceMao == MAO_DIREITA)
				//ajusta a região de interesse (mão) pela movimentação
				if (mao.confidence == 1.0) {
                    if (!maoProximaPerimetro(mao.x, mao.y)) {
                        roi.x = mao.x - ROI_OFFSET;
                        roi.y = mao.y - ROI_OFFSET;
                    }//if (!maoProximaPerimetro(mao.x, mao.y))
                }//if (mao.confidence == 1.0)
				
				//mapas coloridos da mão
				Mat mapaMaoBGR(mapaBGR, roi);
				mapaMaosBGR.push_back(mapaMaoBGR);
				//mapas de profundidade da mão
				Mat mapaMao(mapaProfundidade, roi);
				Mat mapaDisparidadeColorido;

				//coloriza a disparidade encontrada pela câmera de profundidade
				colorizeDisparity(mapaMao, mapaDisparidadeColorido, -1);
				mapaDisparidadeColorido.copyTo(mapaDisparidadeColorido, mapaMao != 0);
				
				//aplica um blur pra tentar conter o ruído
				//cv::Size a(5,5);
				//GaussianBlur(mapaDisparidadeColoridoValido, mapaDisparidadeColoridoValido, a, 1); 
				//blur(mapaDisparidadeColorido, mapaDisparidadeColoridoValido, a);
				medianBlur(mapaDisparidadeColorido, mapaDisparidadeColorido, 5);

				//Descarta a profundidade a partir do OFFSET de cor
				//o elemento mais próximo (possívelmente a mão) e uma pequena distância a partir dele permancem
				int i, j;
				for (i = 0; i < mapaDisparidadeColorido.rows; i++) {
					for (j = 0; j < mapaDisparidadeColorido.cols; j++) {
						if(mapaDisparidadeColorido.at<Vec3b>(i, j)[R] 
						< (mapaDisparidadeColorido.at<Vec3b>(i, j)[B])+COR_OFFSET
						|| mapaDisparidadeColorido.at<Vec3b>(i, j)[R] 
						< (mapaDisparidadeColorido.at<Vec3b>(i, j)[G])+COR_OFFSET) {
							mapaDisparidadeColorido.at<Vec3b>(i, j)[B] = 0;
							mapaDisparidadeColorido.at<Vec3b>(i, j)[G] = 0;
							mapaDisparidadeColorido.at<Vec3b>(i, j)[R] = 0;
						} else if (mapaDisparidadeColorido.at<Vec3b>(i, j)[R] == 255) {
							//printf("\n %d ",mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[G]); 
						}//if(mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[R] ...
					}//for ( j = 0; j < mapaDisparidadeColoridoValido.cols; j++)
				}//for( i = 0; i < mapaDisparidadeColoridoValido.rows; i++)
							

				//converte para cinza para realizar threshold
				Mat mapaThreshold;
				cvtColor(mapaDisparidadeColorido, mapaThreshold, CV_BGR2GRAY);
				//threshold de Otsu, adaptável, indetifica automático o nível adequado
				threshold(mapaThreshold, mapaThreshold, 50, 255, THRESH_BINARY+THRESH_OTSU);
				//busca os contornos, possívelmente interrompidos por ruído
				vector<vector<Point>> contornos;
				findContours(mapaThreshold, contornos, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
				//se há contornos
				if (contornos.size()) {
					//busca o maior contorno (possívelmente a mão)
					int indiceMaior = 0;
					double maiorArea = 0;
					double area = 0;
					for (i = 0; i < contornos.size(); i++) {
						area = contourArea(contornos[i]);
						if (maiorArea < area) {
							indiceMaior = i;
							maiorArea = area;
						}//if (maiorArea > cArea)
					}//for (int i = 0; i < contornos.size(); i++)
					
					//desenha o maior contorno encontrado
					drawContours(mapaDisparidadeColorido, contornos, indiceMaior, COR_BRANCO, 1, 8, noArray(), 0, Point() );
					
					//será busca o ponto mais interno do contorno para obter o maior círculo interno (possívelmente a palma/mão)
					//o objetivo é remover parte do antebraço, devido ao ruído causado por presença ou ausência de diferentes tipos de manga de roupas
					Mat mapaPontoInterno(mapaDisparidadeColorido.size(), CV_32FC1);
					mapaPontoInterno.setTo(0);
					//obtém mapa com as distâncias dos pontos em relação ao contorno
					for (i = 0; i < mapaPontoInterno.rows; i++) {
						for (j = 0;j< mapaPontoInterno.cols;j++) {
							mapaPontoInterno.at<float>(i,j) = pointPolygonTest(contornos[indiceMaior], Point(i,j), true);
						}//for (j = 0;j< dist.cols;j++)
					}//for (i = 0; i < dist.rows; i++)
					//calcular o ponto mais interno do contorno
					int ix = 0, jx = 0;
					float distanciaPonto = -1;
					for (i = 0; i < mapaPontoInterno.rows; i++) {
						for (j = 0; j < mapaPontoInterno.cols; j++) {
							if (mapaPontoInterno.at<float>(i,j) > distanciaPonto) {
								distanciaPonto = mapaPontoInterno.at<float>(i,j);
								ix = i;
								jx = j;
							}//if (dist.at<float>(i,j) > maxdist)
						}//for (j = 0; j < dist.cols; j++)
					}//for (i = 0; i < dist.rows; i++)
					//desenha o ponto e o maior círculo interno  						
					Point centro = Point(ix, jx);
					circle(mapaDisparidadeColorido, centro, 1, COR_AZUL, 1, CV_AA);
					circle(mapaDisparidadeColorido, centro, abs(distanciaPonto), COR_AZUL, 1, CV_AA);					
					//remove o espaço abaixo do círculo, possívelmente o antebraço
					for (i = (jx+distanciaPonto); i < mapaDisparidadeColorido.rows; i++) {
						for (j = 0; j < mapaDisparidadeColorido.cols; j++) {
							mapaDisparidadeColorido.at<Vec3b>(i, j)[B] = 0;
							mapaDisparidadeColorido.at<Vec3b>(i, j)[G] = 0;
							mapaDisparidadeColorido.at<Vec3b>(i, j)[R] = 0;
						}//for ( j = 0; j < mapaDisparidadeColoridoValido.cols; j++)
					}//for( i = 0; i < mapaDisparidadeColoridoValido.rows; i++)
				}//if (contornos.size()) {

				//clona o mapa para poder inverter, no reconhecimento da mão esquerda, visto que o treinamento é com a mão direita
				Mat mapaTemporario = mapaDisparidadeColorido.clone();
				if (indiceMao == MAO_ESQUERDA) {
					flip(mapaTemporario, mapaTemporario, 1);
				}//if (indiceMao == MAO_ESQUERDA)

				//monta um vetor da matriz do mapa, com os valores RGB concatenados e em sequência de três digitos forçada pela soma com o OFFSET (100)
				//será utilizado pelo SVM
				Mat dadosTeste(1, areaMapa, CV_32FC1);
				int k = 0;
				for (i = 0; i < mapaTemporario.rows; i++) {
					for (j = 0; j < mapaTemporario.cols; j++) {
						int tR = mapaTemporario.at<Vec3b>(i, j)[R];
						int tG = mapaTemporario.at<Vec3b>(i, j)[G];
						int tB = mapaTemporario.at<Vec3b>(i, j)[B];
						dadosTeste.at<float>(0, k++) = concatenarNumeros(concatenarNumeros(tR+NUMERO_OFFSET, tG+NUMERO_OFFSET), tB+NUMERO_OFFSET);
					}//for ( j = 0; j < mapaTemporario.cols; j++)
				}//for( i = 0; i < mapaTemporario.rows; i++)
				
				//usa o svm e imprime o resultado
				if (TREINAMENTO == 0) {
					if (indiceMao == MAO_DIREITA) {
						printf("\n\nReconhecimento\nMao direita:");
						printf("\nSVM (1vs1)\nLINEAR: ");
						imprimirResposta(linearSVM.predict(dadosTeste));
						printf("\nPOLINOMIAL: ");
						imprimirResposta(polinomialSVM.predict(dadosTeste));
						printf("\nRADIAL: ");
						imprimirResposta(radialSVM.predict(dadosTeste));
					} else {
						printf("\nMao esquerda: ");
						printf("\nSVM (1vs1)\nLINEAR: ");
						imprimirResposta(linearSVM.predict(dadosTeste));
						printf("\nPOLINOMIAL: ");
						imprimirResposta(polinomialSVM.predict(dadosTeste));
						printf("\nRADIAL: ");
						imprimirResposta(radialSVM.predict(dadosTeste));
					}//if (indiceMao == 1)
					
				}//if (delay == 0)

				//adiciona no vetor de mãos (direita/esquerda)
				mapaMaos.push_back(mapaDisparidadeColorido);
				
				//desenha o esqueleto do torso, braços e cabeça no mapa de profundidade
				cvtColor(mapaProfundidade, mapaProfundidadeEsqueleto, CV_GRAY2BGR);
				//recebe os pontos pelo kinect
				Point torso(esqueleto.torso.x, esqueleto.torso.y);
				Point pescoco(esqueleto.neck.x, esqueleto.neck.y);
				Point cabeca(esqueleto.head.x, esqueleto.head.y);
				Point maoEsq = Point(esqueleto.leftHand.x, esqueleto.leftHand.y);
				Point cotoveloEsq = Point(esqueleto.leftElbow.x, esqueleto.leftElbow.y);
				Point ombroEsq = Point(esqueleto.leftShoulder.x, esqueleto.leftShoulder.y);
				Point maoDir = Point(esqueleto.rightHand.x, esqueleto.rightHand.y);
				Point cotoveloDir = Point(esqueleto.rightElbow.x, esqueleto.rightElbow.y);
				Point ombroDir = Point(esqueleto.rightShoulder.x, esqueleto.rightShoulder.y);
				//desenha a estrutura do braço esquerdo
				circle(mapaProfundidadeEsqueleto, maoEsq, 2, COR_AMARELO, 2);
				circle(mapaProfundidadeEsqueleto, cotoveloEsq, 2, COR_AMARELO, 2);
				circle(mapaProfundidadeEsqueleto, ombroEsq, 2, COR_AMARELO, 2);
				line(mapaProfundidadeEsqueleto, maoEsq, cotoveloEsq, COR_AMARELO, 1);
				line(mapaProfundidadeEsqueleto, cotoveloEsq, ombroEsq, COR_AMARELO, 1);
				//desenha a estrutura do braço direito
				circle(mapaProfundidadeEsqueleto, maoDir, 2, COR_VERDE, 2);
				circle(mapaProfundidadeEsqueleto, cotoveloDir, 2, COR_VERDE, 2);
				circle(mapaProfundidadeEsqueleto, ombroDir, 2, COR_VERDE, 2);
				line(mapaProfundidadeEsqueleto, maoDir, cotoveloDir, COR_VERDE, 1);
				line(mapaProfundidadeEsqueleto, cotoveloDir, ombroDir, COR_VERDE, 1);
				//desenha a estrutura do torso e cabeça
				circle(mapaProfundidadeEsqueleto, torso, 2, COR_BRANCO, 2);
				circle(mapaProfundidadeEsqueleto, pescoco, 2, COR_VERMELHO, 2);
				line(mapaProfundidadeEsqueleto, ombroEsq, torso, COR_BRANCO, 1);
				line(mapaProfundidadeEsqueleto, ombroEsq, pescoco, COR_BRANCO, 1);
				line(mapaProfundidadeEsqueleto, ombroDir, torso, COR_BRANCO, 1);
				line(mapaProfundidadeEsqueleto, ombroDir, pescoco, COR_BRANCO, 1);
				circle(mapaProfundidadeEsqueleto, cabeca, 2, COR_AZUL, 2);
				line(mapaProfundidadeEsqueleto, pescoco, cabeca, COR_VERMELHO, 1);
			}//if (sensor->getNumTrackedUsers() > 0)
        }//for (int indiceMao = 0; indiceMao < 2; indiceMao++)

		//recebe a interação do teclado
		teclado = waitKey(10);

		if (teclado == 'a' || teclado == 'b') {
			if (teclado == 'a' && quantidadeTreinamentoA >= REPETICAO) {
				printf("\nLimite de treinamento da letra A!");
			} else if (teclado == 'b' && quantidadeTreinamentoB >= REPETICAO) {
				printf("\nLimite de treinamento da letra B!");
			}//if (teclado == 'a' && quantidadeTreinamentoA >= REPETICAO)
			//monta um vetor da matriz do mapa, com os valores RGB concatenados e em sequência de três digitos forçada pela soma com o OFFSET (100)
			//será utilizado pelo SVM
			int i, j, k = 0;
			for(i = 0; i < mapaMaos[MAO_DIREITA].rows; i++) {
				for (j = 0; j < mapaMaos[MAO_DIREITA].cols; j++) {
					int tR = mapaMaos[MAO_DIREITA].at<Vec3b>(i, j)[R];
					int tG = mapaMaos[MAO_DIREITA].at<Vec3b>(i, j)[G];
					int tB = mapaMaos[MAO_DIREITA].at<Vec3b>(i, j)[B];
					dadosTreinamento.at<float>(quantidadeTreinamento,k++) = concatenarNumeros(concatenarNumeros(tR+NUMERO_OFFSET, tG+NUMERO_OFFSET), tB+NUMERO_OFFSET);
				}//for (j = 0; j < mapaMaos[0].cols; j++)
			}//for(i = 0; i < mapaMaos[0].rows; i++)

			if (teclado == 'a') {
				printf("\nFrame capturado para treinamento da letra A!");
				label[quantidadeTreinamento] = SINAL_A;
				quantidadeTreinamentoA++;
			} else if (teclado == 'b') {
				printf("\nFrame capturado para treinamento da letra B!");
				label[quantidadeTreinamento] = SINAL_B;
				quantidadeTreinamentoB++;
			} else {
				printf("\nFrame capturado para treinamento de casos errados!");
				label[quantidadeTreinamento] = SINAL_NADA;
			}
			quantidadeTreinamento++;
		} else if (teclado == ' ') {
			if (TREINAMENTO == 0) {
				printf("\nIniciando novo treinamento!");
				TREINAMENTO = 1;
			} else {
				printf("\nIniciando reconhecimento!");
				TREINAMENTO = 0;
			}//if (TREINAMENTO == 0)
		} else if (teclado == 27) {
			break;
		}//if (teclado == 'a' || teclado == 'b')

		//apresenta o mapa de profundidade simples caso não tenha detectado o usuário
		//caso tenha detectado, mostra o mapa de profundidade com o esqueleto
		if (sensor->getNumTrackedUsers() > 0) {
	        imshow(frameProfundidade, mapaProfundidadeEsqueleto);
		} else {
			imshow(frameProfundidade, mapaProfundidade);
		}//if (sensor->getNumTrackedUsers() > 0)

		//apresenta a imagem das mãos
        if (mapaMaos.size() >= 2) {
			resize(mapaMaos[MAO_DIREITA], mapaMaos[MAO_DIREITA], Size(), 3, 3);
            resize(mapaMaos[MAO_ESQUERDA], mapaMaos[MAO_ESQUERDA], Size(), 3, 3);
			resize(mapaMaosBGR[MAO_DIREITA], mapaMaosBGR[MAO_DIREITA], Size(), 3, 3);
            resize(mapaMaosBGR[MAO_ESQUERDA], mapaMaosBGR[MAO_ESQUERDA], Size(), 3, 3);
			imshow(frameMaoDireita, mapaMaos[MAO_DIREITA]);
            imshow(frameMaoEsquerda, mapaMaos[MAO_ESQUERDA]);
			imshow(frameMaoDireitaBGR, mapaMaosBGR[MAO_DIREITA]);
            imshow(frameMaoEsquerdaBGR, mapaMaosBGR[MAO_ESQUERDA]);
            mapaMaos.clear();
			mapaMaosBGR.clear();
        }//if (mapaMaos.size() >= 2)
    }//while (1)

	//salva o treinamento SVM
	if (TREINAMENTO == 1) {
		Mat labelTreinamento(REPETICAO*2, 1, CV_32FC1, label);

		SVMparams.svm_type    = CvSVM::C_SVC;
		SVMparams.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
		
		//treinamento linear
		SVMparams.kernel_type = CvSVM::LINEAR;
		linearSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), SVMparams);
		linearSVM.save(buscarNomeArquivo(CvSVM::LINEAR));
		//treinamento polinomial quadrático
		SVMparams.degree	  = 2;//quadrática
		SVMparams.kernel_type = CvSVM::POLY;
		polinomialSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), SVMparams);
		polinomialSVM.save(buscarNomeArquivo(CvSVM::POLY));
		//treinamento radial
		SVMparams.kernel_type = CvSVM::RBF;
		radialSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), SVMparams);
		radialSVM.save(buscarNomeArquivo(CvSVM::RBF));

		printf("\nGravando treinamento!");
	}//if (TREINAMENTO == 1)

	mapaMaos.clear();
	mapaMaosBGR.clear();

    delete sensor;

    return 0;
}


/*

					
					//vector<Point> curva;
					//approxPolyDP(contorno, curva, 10, true);
					//minEnc

					//printf("\n size %f", curva.size());	
				
					

					/*vector<vector<Point>> corpo(contornos.size());
					vector<vector<int>> corpoI(contornos.size());
					convexHull(contorno, corpo[0], false);
					convexHull(contorno, corpoI[0], false);

					//printf("\n %d ", corpoI.size());
					//drawContours(mapaDisparidadeColoridoValido, corpo, 0, COR_VERDE, 1, 8, vector<Vec4i>(), 0, Point());

					vector<vector<Vec4i>> defeitosConvexos(contornos.size());
					convexityDefects(Mat(contornos[indiceMaior]), corpoI[0], defeitosConvexos[0]);

					for (j = 0; j < corpo[0].size(); j++)
                    {
						if (corpo[0][j].y <= centro.y) {
							circle(mapaDisparidadeColoridoValido, corpo[0][j], 3, COR_AMARELO, 2);
						}
                    }

					for (j = 0; j < defeitosConvexos[0].size(); j++) {
						//verifica se o defeito esta na altura ou mais alto que o centro da mão (pegar apenas espaço entre dedos)
						if (contorno[defeitosConvexos[0][j].val[2]].y <= centro.y) {
							if (defeitosConvexos[0][j].val[3] >= abs(maxdist)) {
								circle(mapaDisparidadeColoridoValido, contorno[defeitosConvexos[0][j].val[2]], 3, COR_VERDE, 2);
							}
						}
					}*/