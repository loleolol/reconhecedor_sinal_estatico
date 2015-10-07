#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv/cv.h>
#include "SkeletonSensor.hpp"

using namespace std;
using namespace cv;

//define a quantidade de repetições para coletas
const unsigned int REPETICAO = 30;
const unsigned int QUANTIDADE = 570;
//define os sinais reconhecidos
const float SINAL_A = 1.0;
const float SINAL_B = 2.0;
const float SINAL_C = 3.0;
const float SINAL_D = 4.0;
const float SINAL_E = 5.0;
const float SINAL_F = 6.0;
const float SINAL_G = 7.0;
const float SINAL_I = 8.0;
const float SINAL_L = 9.0;
const float SINAL_M = 10.0;
const float SINAL_N = 11.0;
const float SINAL_O = 12.0;
const float SINAL_P = 13.0;
const float SINAL_Q = 14.0;
const float SINAL_R = 15.0;
const float SINAL_S = 16.0;
const float SINAL_T = 17.0;
const float SINAL_U = 18.0;
const float SINAL_V = 19.0;

//define a resolucao do kinect
const unsigned int XRES = 640;
const unsigned int YRES = 480;
//define o valor para extração da ROI (Region of Interest)
const unsigned int ROI_OFFSET = 70;
const unsigned int ROIXRES = 180;
const unsigned int ROIYRES = 180;
//define o valor para extração da coloração produzida pela disparidade
const unsigned int COR_OFFSET = 150;
//define o valor para equalizar números em 3 casas d
const unsigned int NUMERO_OFFSET = 100;
//define o índice das duas mãos
const unsigned int MAO_ESQUERDA = 1;
const unsigned int MAO_DIREITA = 0;
//define os índices das escalas BGR
const unsigned int B = 0;
const unsigned int G = 1;
const unsigned int R = 2;
//define cores
const Scalar COR_VERDE    = Scalar(0,255,0);
const Scalar COR_AZUL     = Scalar(240,40,0);
const Scalar COR_AMARELO  = Scalar(0,128,200);
const Scalar COR_VERMELHO = Scalar(0,0,255);
const Scalar COR_BRANCO   = Scalar(255,255,255);
const Scalar COR_PRETO    = Scalar(0,0,0);
//indices da matriz de dados
const int MAO = 0;
const int TEMPO_EXTRACAO = 1;
const int RESPOSTA_LINEAR = 2;
const int TEMPO_LINEAR = 3;
const int RESPOSTA_POLINOMIAL = 4;
const int TEMPO_POLINOMIAL = 5;
const int RESPOSTA_RADIAL = 6;
const int TEMPO_RADIAL = 7;
//vetor de letras
char letra[19] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'};
char mao[2] = {'D', 'E'};
//frequencia para registro dos timers
LARGE_INTEGER frequencia;

//coloriza a disparidade obtida com câmera de profundidade - função retirada de exemplo do OpenCV
static void colorizeDisparity(const Mat& gray, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f)
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
			if (d != 0) {
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
}

//retorna true se a mão estiver se aproximando do perímetro do frame
bool maoProximaPerimetro(float x, float y)
{
    return (x > (XRES - ROI_OFFSET)) || (x < (ROI_OFFSET)) 
		|| (y > (YRES - ROI_OFFSET)) || (y < (ROI_OFFSET));
}//bool maoProximaPerimetro(float x, float y)

//concatena dois números
unsigned concatenarNumeros(unsigned a, unsigned b) 
{
    unsigned c = 10;
    while (b >= c) {
        c *= 10;
	}//while (b >= c)
    return a * c + b;        
}//unsigned concatenarNumeros(unsigned a, unsigned b)

//retorna o sinal pela tecla
float retornarSinalPorTecla(int tecla)
{
	if (tecla == 'a') {
		return SINAL_A;
	} else if (tecla == 'b') {
		return SINAL_B;
	} else if (tecla == 'c') {
		return SINAL_C;
	} else if (tecla == 'd') {
		return SINAL_D;
	} else if (tecla == 'e') {
		return SINAL_E;
	} else if (tecla == 'f') {
		return SINAL_F;
	} else if (tecla == 'g') {
		return SINAL_G;
	} else if (tecla == 'i') {
		return SINAL_I;
	} else if (tecla == 'l') {
		return SINAL_L;
	} else if (tecla == 'm') {
		return SINAL_M;
	} else if (tecla == 'n') {
		return SINAL_N;
	} else if (tecla == 'o') {
		return SINAL_O;
	} else if (tecla == 'p') {
		return SINAL_P;
	} else if (tecla == 'q') {
		return SINAL_Q;
	} else if (tecla == 'r') {
		return SINAL_R;
	} else if (tecla == 's') {
		return SINAL_S;
	} else if (tecla == 't') {
		return SINAL_T;
	} else if (tecla == 'u') {
		return SINAL_U;
	} else if (tecla == 'v') {
		return SINAL_V;
	} else {
		return 0;
	}//if (tecla == 'a')
}//float retornarSinalPorTecla(int tecla)

//imprime a resposta conforme resultado
void imprimirResposta(double resposta, double tempo)
{
	//int teste = (resposta >= 0.0f ? floorf(resposta + 0.5f) : ceilf(resposta - 0.5f));
	printf("Tempo %fms - LETRA "+letra[(int)resposta], tempo);
	printf(" (%f)", resposta);
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

//mede o tempo decorrido em milisegundos entre os dois marcadores
double medirTempoDecorrido(LARGE_INTEGER ta, LARGE_INTEGER tb)
{
	return (tb.QuadPart - ta.QuadPart) * 1000.0 / frequencia.QuadPart;
}//double medirTempoDecorrido(LARGE_INTEGER ta, LARGE_INTEGER tb)

int main(int argc, char** argv)
{
	//define o classificador SVM (Support Vector Machines) do OpenCV (1-vs-1)
	CvSVM linearSVM;
	CvSVM polinomialSVM;
	CvSVM radialSVM;
	CvSVMParams linearSVMparams;
	linearSVMparams.svm_type    = CvSVM::C_SVC;
	linearSVMparams.kernel_type = CvSVM::LINEAR;
	CvSVMParams polinomialSVMparams;
	polinomialSVMparams.svm_type = CvSVM::C_SVC;
	polinomialSVMparams.kernel_type = CvSVM::POLY;
	polinomialSVMparams.degree = 2;
	CvSVMParams radialSVMparams;
	radialSVMparams.svm_type     = CvSVM::C_SVC;
	radialSVMparams.kernel_type = CvSVM::RBF;
	
	//calcular o tempo decorrido em milisegundos
	LARGE_INTEGER t1, t2, t3, t4, t5, t6, t7, t8;

	//TODO
	Ptr<DescriptorMatcher> comparador(new FlannBasedMatcher);
	Ptr<FeatureDetector> detetor(new SurfFeatureDetector(40, 4, 2, true, true));//Hessian threshold (40), considera direção
	Ptr<DescriptorExtractor> extrator = new SurfDescriptorExtractor(40, 4, 2, true, true);//Hessian threshold (40), considera direção
	BOWImgDescriptorExtractor dextrator(extrator, comparador);
	Mat caracteristicasDesagrupadas;

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

	Mat dadosTreinamentoSURF;
	Mat labelTreinamento;
	double dados[(QUANTIDADE*2)][8];
	int quantidadeTreinamento[19] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int quantidadeReconhecimento = 0;
	int quantidadeReconhecimentoEtapa = 0;
	char letra[19] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'};

	Point palma[2] = {Point(0, 0), Point(0, 0)};
	int raioPalma[2] = {0, 0};

	//cria os frames que serão usados
    namedWindow(frameProfundidade, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoEsquerda, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoDireita, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoEsquerdaBGR, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoDireitaBGR, CV_WINDOW_AUTOSIZE);

	//interação com o teclado
	int teclado = 0;
	
    Mat vocabulario; 
    FileStorage fs("dicionario.xml", FileStorage::READ);
    fs["vocabulario"] >> vocabulario;
    fs.release();

	dextrator.setVocabulary(vocabulario);
	printf("\nVocabulario existente carregado!");

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
				//seta a frequencia
				QueryPerformanceFrequency(&frequencia);
				//inicia o timer pra contabilizar o tempo da extração de características
				QueryPerformanceCounter(&t1);
				//obtém o esqueleto
                Skeleton esqueleto =  sensor->getSkeleton(sensor->getUID(0));
                SkeletonPoint mao, cotovelo;
				//define a mão com base no índice
                if (indiceMao == MAO_DIREITA) {
					//direita para o usuário
                    mao = esqueleto.leftHand;
					cotovelo = esqueleto.leftElbow;
				} else {
					//esquerda para o usuário
					mao = esqueleto.rightHand;
					cotovelo = esqueleto.rightElbow;
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

				//desenha o retângulo da região de interesse para referência
				rectangle(mapaProfundidade, Point(roi.x-1, roi.y-1), Point(roi.x+ROIXRES+1, roi.y+ROIYRES+1), COR_BRANCO, 1);

				//coloriza a disparidade encontrada pela câmera de profundidade
				colorizeDisparity(mapaMao, mapaDisparidadeColorido, -1);
				mapaDisparidadeColorido.copyTo(mapaDisparidadeColorido, mapaMao != 0);
				
				//aplica um blur pra tentar conter o ruído
				medianBlur(mapaDisparidadeColorido, mapaDisparidadeColorido, 5);

				//Descarta a profundidade a partir do OFFSET de cor
				//o elemento mais próximo (possívelmente a mão) e uma pequena distância a partir dele permancem
				int i, j, k = 0;
				for (i = 0; i < mapaDisparidadeColorido.rows; i++) {
					for (j = 0; j < mapaDisparidadeColorido.cols; j++) {
						if(mapaDisparidadeColorido.at<Vec3b>(i, j)[R] 
						< (mapaDisparidadeColorido.at<Vec3b>(i, j)[B])+COR_OFFSET
						|| mapaDisparidadeColorido.at<Vec3b>(i, j)[R] 
						< (mapaDisparidadeColorido.at<Vec3b>(i, j)[G])+COR_OFFSET) {
							mapaDisparidadeColorido.at<Vec3b>(i, j)[B] = 0;
							mapaDisparidadeColorido.at<Vec3b>(i, j)[G] = 0;
							mapaDisparidadeColorido.at<Vec3b>(i, j)[R] = 0;
						}//if(mapaDisparidadeColoridoValido.at<Vec3b>(i, j)[R] ...
					}//for ( j = 0; j < mapaDisparidadeColoridoValido.cols; j++)
				}//for( i = 0; i < mapaDisparidadeColoridoValido.rows; i++)
				
				//converte para cinza para realizar detecção de borda Canny
				Mat mapaCinza, mapaThreshold, mapaCanny;
				cvtColor(mapaDisparidadeColorido, mapaCinza, CV_BGR2GRAY);

				//detectação de bordas Canny para ressaltar características internas da mão, threshold baixo (20 ~ 40) para detectar mais características
				Canny(mapaCinza, mapaCanny, 20, 20*2);	
				vector<vector<Point>> contornos;
				findContours(mapaCanny, contornos, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
				for (i = 0; i < contornos.size(); i++) {
					drawContours(mapaDisparidadeColorido, contornos, i, COR_AMARELO, 1, 8, noArray(), 0, Point() );
				}//for (i = 0; i < contornosCanny.size(); i++)

				//converte para cinza para realizar threshold e detectar a borda externa
				cvtColor(mapaDisparidadeColorido, mapaCinza, CV_BGR2GRAY);
				
				//threshold de Otsu, adaptável, indetifica automático o nível adequado
				threshold(mapaCinza, mapaThreshold, 50, 255, THRESH_BINARY+THRESH_OTSU);
				
				//busca os contornos
				findContours(mapaThreshold, contornos, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

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
				
					//será buscado o ponto mais interno do contorno para obter o maior círculo interno (possívelmente a palma/mão)
					//o objetivo é remover parte do antebraço, devido ao ruído causado por presença ou ausência de diferentes tipos de manga de roupas
					Mat mapaPontoInterno(mapaDisparidadeColorido.size(), CV_32FC1);
					mapaPontoInterno.setTo(0);

					//obtém mapa com as distâncias dos pontos em relação ao contorno
					for (i = 0; i < mapaPontoInterno.rows; i++) {
						for (j = 0; j < mapaPontoInterno.cols; j++) {
							mapaPontoInterno.at<float>(i,j) = pointPolygonTest(contornos[indiceMaior], Point(i,j), true);
						}//for (j = 0;j< dist.cols;j++)
					}//for (i = 0; i < dist.rows; i++)
					
					//busca o ponto médio do polígono para auxiliar na referência de busca do círculo
					//acaba servindo caso a pessoa esteja vestindo casaco grande (círculo do braço maior que círculo da palma)
					double pontoMedioY = ROIYRES;
                    Scalar media = mean(contornos[indiceMaior]);
					if (typeid(media) == typeid(Scalar)) {
						pontoMedioY = media.val[1];
					}//if (typeid(media) == typeid(Scalar))
                    //Point pontoMedio = Point(media.val[0], media.val[1]);
					//circle(mapaDisparidadeColorido, centerPoint, 1, COR_VERDE, 1, CV_AA);

					//calcular o ponto mais interno do contorno
					int ix = 0, jx = 0;
					float distanciaPonto = -1;
					for (i = 0; i < mapaPontoInterno.rows; i++) {
						for (j = 0; j < mapaPontoInterno.cols; j++) {
							if (mapaPontoInterno.at<float>(i,j) > distanciaPonto) {
								//evita encontrar círculos afastados do ponto médio, próximos da borda (antebraço
								if (pontoMedioY >= (j-abs(mapaPontoInterno.at<float>(i,j)))) {
									distanciaPonto = mapaPontoInterno.at<float>(i,j);
									ix = i;
									jx = j;
								}//if (pontoMedioY >= (j-abs(mapaPontoInterno.at<float>(i,j))))
							}//if (mapaPontoInterno.at<float>(i,j) > distanciaPonto)
						}//for (j = 0; j < dist.cols; j++)
					}//for (i = 0; i < dist.rows; i++)
										
					//desenha o ponto e o maior círculo interno
					if (distanciaPonto > -1) {
						palma[indiceMao] = Point(roi.x+ix, roi.y+jx);
						raioPalma[indiceMao] = abs(distanciaPonto);
						//circle(mapaDisparidadeColorido, Point(ix, jx), 1, COR_AZUL, 1, CV_AA);
						//circle(mapaDisparidadeColorido, Point(ix, jx), abs(distanciaPonto), COR_AZUL, 1, CV_AA);
						//só corta o cotovelo se ele estiver pra baixo (tentar resolver para as letras m e n)
						if (abs(cotovelo.y) > abs(mao.y)) {
							//remove o espaço abaixo do círculo, possívelmente o antebraço
							for (i = (jx+distanciaPonto); i < mapaDisparidadeColorido.rows; i++) {
								for (j = 0; j < mapaDisparidadeColorido.cols; j++) {
									if (mapaDisparidadeColorido.at<Vec3b>(i, j)[R] != 0) {
										mapaDisparidadeColorido.at<Vec3b>(i, j)[B] = 0;
										mapaDisparidadeColorido.at<Vec3b>(i, j)[G] = 0;
										mapaDisparidadeColorido.at<Vec3b>(i, j)[R] = 0;
									}//if (mapaDisparidadeColorido.at<Vec3b>(i, j)[R] != 0)
								}//for ( j = 0; j < mapaDisparidadeColoridoValido.cols; j++)
							}//for( i = 0; i < mapaDisparidadeColoridoValido.rows; i++)
						}//if (cotovelo.y < (jx+distanciaPonto))
					}//if (distanciaPonto > -1)

					//após toda a extração de características, a idéia é fechar o contorno externo da mão
					//converte para cinza para realizar threshold e detectar a borda externa
					cvtColor(mapaDisparidadeColorido, mapaCinza, CV_BGR2GRAY);				
					//threshold de Otsu, adaptável, indetifica automático o nível adequado
					threshold(mapaCinza, mapaThreshold, 50, 255, THRESH_BINARY+THRESH_OTSU);
					//busca os contornos
					findContours(mapaThreshold, contornos, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

					//desenha os contornos encontrado
					for (i = 0; i < contornos.size(); i++) {
						drawContours(mapaDisparidadeColorido, contornos, i, COR_BRANCO, 1, 8, noArray(), 0, Point() );
					}//for (i = 0; i < contornosCanny.size(); i++)
				}//if (contornos.size()) {

				//clona o mapa para poder inverter, no reconhecimento da mão esquerda, visto que o treinamento é com a mão direita
				Mat mapaTemporario = mapaDisparidadeColorido.clone();
				if (indiceMao == MAO_ESQUERDA) {
					flip(mapaTemporario, mapaTemporario, 1);
				}//if (indiceMao == MAO_ESQUERDA)
				
				//extrai os descritores com o SURF considerando o vocabulário da Bag-of-Words 
				Mat descritor;
				vector<KeyPoint> pontosChave;
				detetor->detect(mapaTemporario, pontosChave);
				dextrator.compute(mapaTemporario, pontosChave, descritor);
				
				//pega o timer do fim da extração e computa o tempo decorrido
				QueryPerformanceCounter(&t2);
				//usa o svm e imprime o resultado
				if (TREINAMENTO == 0) {
					if (quantidadeReconhecimentoEtapa < (REPETICAO*2)) {
						dados[quantidadeReconhecimento][MAO] = indiceMao;
						dados[quantidadeReconhecimento][TEMPO_EXTRACAO] = medirTempoDecorrido(t1, t2);

						if (indiceMao == MAO_DIREITA) {
							printf("\n\nEXTRACAO DE CARACTERISTICAS %fms\nRECONHECIMENTO MAO DIREITA:", dados[quantidadeReconhecimento][TEMPO_EXTRACAO]);
						} else {
							printf("\nEXTRACAO DE CARACTERISTICAS %fms\nRECONHECIMENTO MAO ESQUERDA:", dados[quantidadeReconhecimento][TEMPO_EXTRACAO]);
						}//if (indiceMao == 1)

						//predição linear, contabiliza o tempo
						printf("\nSVM (1vs1) LINEAR: ");
						QueryPerformanceCounter(&t3);
						dados[quantidadeReconhecimento][RESPOSTA_LINEAR] = linearSVM.predict(descritor);
						QueryPerformanceCounter(&t4);
						dados[quantidadeReconhecimento][TEMPO_LINEAR] = medirTempoDecorrido(t3, t4);
						imprimirResposta(dados[quantidadeReconhecimento][RESPOSTA_LINEAR], dados[quantidadeReconhecimento][TEMPO_LINEAR]);
						//predição polinomial, contabiliza o tempo
						printf("\nSVM (1vs1) POLINOMIAL: ");
						QueryPerformanceCounter(&t5);
						dados[quantidadeReconhecimento][RESPOSTA_POLINOMIAL] = polinomialSVM.predict(descritor);
						QueryPerformanceCounter(&t6);
						dados[quantidadeReconhecimento][TEMPO_POLINOMIAL] = medirTempoDecorrido(t5, t6);
						imprimirResposta(dados[quantidadeReconhecimento][RESPOSTA_POLINOMIAL], dados[quantidadeReconhecimento][TEMPO_POLINOMIAL]);
						//predição radial, contabiliza o tempo
						printf("\nSVM (1vs1) RADIAL: ");
						QueryPerformanceCounter(&t7);
						dados[quantidadeReconhecimento][RESPOSTA_RADIAL] = radialSVM.predict(descritor);
						QueryPerformanceCounter(&t8);
						dados[quantidadeReconhecimento][TEMPO_RADIAL] = medirTempoDecorrido(t7, t8);
						imprimirResposta(dados[quantidadeReconhecimento][RESPOSTA_RADIAL], dados[quantidadeReconhecimento][TEMPO_RADIAL]);
					
						quantidadeReconhecimento++;
						quantidadeReconhecimentoEtapa++;
					} else {

						printf("\n Completou uma etapa de %d reconhecimentos!", quantidadeReconhecimento);
						quantidadeReconhecimentoEtapa = 0;
						TREINAMENTO = 1;

					}//if (quantidadeReconhecimento < QUANTIDADE)
				}//if (TREINAMENTO == 0)*/

				//adiciona no vetor de mãos (direita/esquerda)
				mapaMaos.push_back(mapaDisparidadeColorido);
				
				//desenha o esqueleto do torso, braços e cabeça no mapa de profundidade colorido, apenas para referência, não contabiliza o tempo
				//cria o mapa de profundidade colorido, para observar depuração dos esqueleto e da ROI
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
				circle(mapaProfundidadeEsqueleto, pescoco, 2, COR_AZUL, 2);
				line(mapaProfundidadeEsqueleto, ombroEsq, torso, COR_BRANCO, 1);
				line(mapaProfundidadeEsqueleto, ombroEsq, pescoco, COR_BRANCO, 1);
				line(mapaProfundidadeEsqueleto, ombroDir, torso, COR_BRANCO, 1);
				line(mapaProfundidadeEsqueleto, ombroDir, pescoco, COR_BRANCO, 1);
				circle(mapaProfundidadeEsqueleto, cabeca, 2, COR_AZUL, 2);
				line(mapaProfundidadeEsqueleto, pescoco, cabeca, COR_AZUL, 1);
				//desenha o círculo interno (possívelmente a palma) no mapa do esqueleto
				circle(mapaProfundidadeEsqueleto, palma[MAO_DIREITA], 1, COR_VERMELHO, 1);
				circle(mapaProfundidadeEsqueleto, palma[MAO_DIREITA], raioPalma[MAO_DIREITA], COR_VERMELHO, 1);
				circle(mapaProfundidadeEsqueleto, palma[MAO_ESQUERDA], 1, COR_VERMELHO, 1);
				circle(mapaProfundidadeEsqueleto, palma[MAO_ESQUERDA], raioPalma[MAO_ESQUERDA], COR_VERMELHO, 1);
			}//if (sensor->getNumTrackedUsers() > 0)
        }//for (int indiceMao = 0; indiceMao < 2; indiceMao++)

		//recebe a interação do teclado
		int tecla = waitKey(10);
		if (tecla > 0) {
			teclado = tecla;
		}//if (tecla > 0)
		
		int sinal = (int)retornarSinalPorTecla(teclado);
		if (sinal != 0) {
			if (quantidadeTreinamento[sinal] >= REPETICAO) {
				printf("\nLimite de treinamento da letra %c!", teclado);
				teclado = 999;
			} else {
				//extrai os descritores SURF da imagem, considerando o vocabulário da Bag-of-Words
				Mat descritoresTreinamento;
				vector<KeyPoint> pontosChaveTreinamento;
				detetor->detect(mapaMaos[MAO_DIREITA], pontosChaveTreinamento);
				dextrator.compute(mapaMaos[MAO_DIREITA], pontosChaveTreinamento, descritoresTreinamento);
				dadosTreinamentoSURF.push_back(descritoresTreinamento);
								
				labelTreinamento.push_back(sinal);
				quantidadeTreinamento[sinal]++;

				printf("\nFrame capturado para treinamento da letra %c!", teclado);
			}//if (quantidadeTreinamento[sinal] >= REPETICAO)

		} else if (teclado == ' ') {

			if (TREINAMENTO == 1) {
				printf("\nIniciando reconhecimento!");
				TREINAMENTO = 0;
				teclado = 999;
			}//if (TREINAMENTO == 0)

		} else if (teclado == 'z') {

			for (int i = 0; i < dadosTreinamentoSURF.rows; i++) {
				Mat descritores;
				vector<KeyPoint> pontosChave;
				detetor->detect(dadosTreinamentoSURF.at<Mat>(i), pontosChave);
				extrator->compute(dadosTreinamentoSURF.at<Mat>(i), pontosChave, descritores);
				caracteristicasDesagrupadas.push_back(descritores);
				printf("\nDescritores SURF capturados! (%d)", descritores.size().height);
			}//for (int i = 0; i < dadosTreinamentoSURF.rows; i++)

			BOWKMeansTrainer bagOfWords(caracteristicasDesagrupadas.size().height, TermCriteria(CV_TERMCRIT_ITER,100,0.0001), 1, KMEANS_PP_CENTERS);
			vocabulario = bagOfWords.cluster(caracteristicasDesagrupadas);
			dextrator.setVocabulary(vocabulario);
			printf("\nVocabulario carregado!");
			FileStorage fs("dicionario.xml", FileStorage::WRITE);
			fs << "vocabulario" << vocabulario;
			fs.release();
			printf("\nVocabulario salvo!");

			printf("\n Dados %d - Labels %d", dadosTreinamentoSURF.rows, labelTreinamento.rows);
			//salva o treinamento SVM
			//treinamento linear
			linearSVM.train_auto(dadosTreinamentoSURF, labelTreinamento, Mat(), Mat(), linearSVMparams);
			linearSVM.save(buscarNomeArquivo(CvSVM::LINEAR));
			///treinamento polinomial quadrático
			polinomialSVM.train_auto(dadosTreinamentoSURF, labelTreinamento, Mat(), Mat(), polinomialSVMparams);
			polinomialSVM.save(buscarNomeArquivo(CvSVM::POLY));
			//treinamento radial
			radialSVM.train_auto(dadosTreinamentoSURF, labelTreinamento, Mat(), Mat(), radialSVMparams);
			radialSVM.save(buscarNomeArquivo(CvSVM::RBF));
			printf("\nGravando treinamento!");
			teclado = 999;

		} else if (teclado == 27) {
			ofstream fs("resultado.csv");
			for (int i = 0; i < (QUANTIDADE*2); i++) {				
				fs << mao[(int)dados[i][MAO]] << ";" << dados[i][TEMPO_EXTRACAO]
				<< ";" << letra[(int)dados[i][RESPOSTA_LINEAR]] << ";" << dados[i][TEMPO_LINEAR]
				<< ";" << letra[(int)dados[i][RESPOSTA_POLINOMIAL]] << ";" << dados[i][TEMPO_POLINOMIAL]
				<< ";" << letra[(int)dados[i][RESPOSTA_RADIAL]] << ";" << dados[i][TEMPO_RADIAL] << endl;
			}//for (int i = 0; i < QUANTIDADE; i++)
			fs.close();
			break;

		}//if (sinal != 0)

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

	delete sensor;

	mapaMaos.clear();
	mapaMaosBGR.clear();

    return 0;
}