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
const unsigned int REPETICAO = 40;
const unsigned int REPETICAOETAPA = 40;
const unsigned int QUANTIDADE = 4000;
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
const float SINAL_W = 20.0;

//define a resolucao do kinect
const unsigned int XRES = 640;
const unsigned int YRES = 480;
//define o valor para extração da ROI (Region of Interest)
const unsigned int ROI_OFFSET = 90;
const unsigned int ROIXRES = 180;
const unsigned int ROIYRES = 180;
//define o valor para extração da coloração produzida pela disparidade
const unsigned int COR_OFFSET = 150;
//define o valor para equalizar números em 3 casas d
const unsigned int NUMERO_OFFSET = 100;
//define os índices das escalas BGR
const unsigned int B = 0;
const unsigned int G = 1;
const unsigned int R = 2;
//define cores
const Scalar COR_VERDE    = Scalar(0,255,0);
const Scalar COR_AZUL     = Scalar(255,255,0);
const Scalar COR_AMARELO  = Scalar(0,255,255);
const Scalar COR_VERMELHO = Scalar(0,0,255);
const Scalar COR_BRANCO   = Scalar(255,255,255);
const Scalar COR_PRETO    = Scalar(0,0,0);
//indices da matriz de dados
const int TEMPO_EXTRACAO = 0;
const int RESPOSTA = 1;
const int TEMPO = 2;
//vetor de letras
char letra[21] = {'.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W'};
//respostas
double dados[REPETICAOETAPA*20][3];

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
	} else if (tecla == 'w') {
		return SINAL_W;
	} else {
		return 0;
	}//if (tecla == 'a')
}//float retornarSinalPorTecla(int tecla)

float retornarSinalPorQuantidade(int indice) 
{
	if (indice < REPETICAOETAPA) {
		return SINAL_A;
	} else if (indice < (REPETICAOETAPA*2)) {
		return SINAL_B;
	} else if (indice < (REPETICAOETAPA*3)) {
		return SINAL_C;
	} else if (indice < (REPETICAOETAPA*4)) {
		return SINAL_D;
	} else if (indice < (REPETICAOETAPA*5)) {
		return SINAL_E;
	} else if (indice < (REPETICAOETAPA*6)) {
		return SINAL_F;
	} else if (indice < (REPETICAOETAPA*7)) {
		return SINAL_G;
	} else if (indice < (REPETICAOETAPA*8)) {
		return SINAL_I;
	} else if (indice < (REPETICAOETAPA*9)) {
		return SINAL_L;
	} else if (indice < (REPETICAOETAPA*10)) {
		return SINAL_M;
	} else if (indice < (REPETICAOETAPA*11)) {
		return SINAL_N;
	} else if (indice < (REPETICAOETAPA*12)) {
		return SINAL_O;
	} else if (indice < (REPETICAOETAPA*13)) {
		return SINAL_P;
	} else if (indice < (REPETICAOETAPA*14)) {
		return SINAL_Q;
	} else if (indice < (REPETICAOETAPA*15)) {
		return SINAL_R;
	} else if (indice < (REPETICAOETAPA*16)) {
		return SINAL_S;
	} else if (indice < (REPETICAOETAPA*17)) {
		return SINAL_T;
	} else if (indice < (REPETICAOETAPA*18)) {
		return SINAL_U;
	} else if (indice < (REPETICAOETAPA*19)) {
		return SINAL_V;
	} else {
		return SINAL_W;
	}//if (indice < REPETICAO)
}//float retornarSinalPorQuantidade(int indice) 

//imprime a resposta conforme resultado
void imprimirResposta(double resposta, double tempo)
{
	//int teste = (resposta >= 0.0f ? floorf(resposta + 0.5f) : ceilf(resposta - 0.5f));
	printf("Tempo %fms - LETRA %c (%f)", tempo, letra[(int)resposta], resposta);
}//void imprimirResposta(float resposta)

//mede o tempo decorrido em milisegundos entre os dois marcadores
double medirTempoDecorrido(LARGE_INTEGER ta, LARGE_INTEGER tb)
{
	return (tb.QuadPart - ta.QuadPart) * 1000.0 / frequencia.QuadPart;
}//double medirTempoDecorrido(LARGE_INTEGER ta, LARGE_INTEGER tb)

void treinarEtapa(string distancia, Mat &treinamento, Mat &label, Ptr<FeatureDetector> detetor, BOWImgDescriptorExtractor dextrator)
{
	int maximo = REPETICAOETAPA*20;
	for (int i = 0; i < maximo; i++) {
		stringstream ss;
		string name = "D:/Leo/Documents/tcc/reconhecedor_sinal_estatico/reconhecedor_sinal_estatico/treinamento";
		string name2 = "m/treinamento_";
		string type = ".png";
		ss << name << distancia << name2 << (i+1) << type;
		string file = ss.str();
		Mat imagem = imread(file, CV_LOAD_IMAGE_UNCHANGED);
		//extrai os descritores SURF da imagem, considerando o vocabulário da Bag-of-Words
		Mat descritoresTreinamento;
		vector<KeyPoint> pontosChaveTreinamento;
		detetor->detect(imagem, pontosChaveTreinamento);
		dextrator.compute(imagem, pontosChaveTreinamento, descritoresTreinamento);

		if (descritoresTreinamento.rows > 0) {
			treinamento.push_back(descritoresTreinamento);
			label.push_back(retornarSinalPorQuantidade(i));
		}//if (descritoresTreinamento.rows > 0)
	}//for (int i = 0; i < QUANTIDADE; i++)
}//void treinarEtapa(string distancia, Mat treinamento, Mat label, Ptr<FeatureDetector> detetor, BOWImgDescriptorExtractor dextrator)

void validarEtapa(string distancia, CvSVM &svm, Ptr<FeatureDetector> detetor, BOWImgDescriptorExtractor dextrator)
{
	LARGE_INTEGER ta, tb;
	int maximo = REPETICAOETAPA*20;
	for (int i = 0; i < maximo; i++) {
		stringstream ss;
		string name = "D:/Leo/Documents/tcc/reconhecedor_sinal_estatico/reconhecedor_sinal_estatico/";
		string type = ".png";
		ss << name << distancia << (i+1) << type;
		string file = ss.str();
		Mat imagem = imread(file, CV_LOAD_IMAGE_UNCHANGED);
		//extrai os descritores SURF da imagem, considerando o vocabulário da Bag-of-Words
		Mat descritores;
		vector<KeyPoint> pontosChave;
		detetor->detect(imagem, pontosChave);
		dextrator.compute(imagem, pontosChave, descritores);

		//mede o tempo decorrido para extração de características
		dados[i][TEMPO_EXTRACAO] = 0;
		printf("\n\nEXTRACAO DE CARACTERISTICAS %fms\nRECONHECIMENTO:", dados[i][TEMPO_EXTRACAO]);

		//predição, contabiliza o tempo
		printf("\nSVM: ");
		QueryPerformanceFrequency(&frequencia);
		QueryPerformanceCounter(&ta);
		dados[i][RESPOSTA] = svm.predict(descritores);
		QueryPerformanceCounter(&tb);
		dados[i][TEMPO] = medirTempoDecorrido(ta, tb);
		imprimirResposta(dados[i][RESPOSTA], dados[i][TEMPO]);
	}//for (int i = 0; i < QUANTIDADE; i++)
}//void treinarEtapa(string distancia, Mat treinamento, Mat label, Ptr<FeatureDetector> detetor, BOWImgDescriptorExtractor dextrator)

int main(int argc, char** argv)
{
	//define o modo de operação padrão
	int TREINAMENTO = 1;

	//define os classificadores SVM (Support Vector Machines) do OpenCV
	CvSVM linearSVM;
	CvSVM polinomialSVM;
	CvSVM radialSVM;
	CvSVMParams linearSVMparams;
	linearSVMparams.svm_type = CvSVM::C_SVC;
	linearSVMparams.kernel_type = CvSVM::LINEAR;
	CvSVMParams polinomialSVMparams;
	polinomialSVMparams.svm_type = CvSVM::C_SVC;
	polinomialSVMparams.kernel_type = CvSVM::POLY;
	CvSVMParams radialSVMparams;
	radialSVMparams.svm_type = CvSVM::C_SVC;
	radialSVMparams.kernel_type = CvSVM::RBF;
	
	//Inicializa SURF e Bag-of-Words
	Ptr<DescriptorMatcher> comparador(new FlannBasedMatcher);//não é utilizado
	Ptr<FeatureDetector> detetor(new SurfFeatureDetector(40, 4, 2, true));//Hessian threshold (40), considera direção
	Ptr<DescriptorExtractor> extrator = new SurfDescriptorExtractor(40, 4, 2, true);//Hessian threshold (40), considera direção
	BOWImgDescriptorExtractor dextrator(extrator, comparador);
	Mat caracteristicasDesagrupadas;

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
	Mat mapaBGR(YRES, XRES, CV_32FC1);

    //retângulo para extração da região de interesse (mão)
    Rect roi;
    roi.width  = ROI_OFFSET*2;
    roi.height = ROI_OFFSET*2;

    //mapas de imagem da mão
    Mat mapaMao, mapaMaoBGR, mapaSURF;
	//matrizes de vocabulario e treinamento
	Mat vocabulario, dadosTreinamento, labelTreinamento;
	vector<Mat> armazenador;
	//guarda a palma e o seu raio
	Point palma = Point(0, 0);
	int raioPalma = 0;

	//calcular o tempo decorrido em milisegundos
	LARGE_INTEGER t1, t2, t3, t4, t5, t6, t7, t8;

	//interação com o teclado
	int teclado = 0;

	//contadores	
	int quantidadeTreinamento[21] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int quantidadeVocabulario = 0;
	int quantidadeReconhecimento = 0;
	int quantidadeReconhecimentoEtapa = 0;

	//cria os frames que serão usados
	string frameProfundidade = "Profundidade com Esqueleto";
	string frameMao = "Referencia Colorizada";
	string frameSURF = "Referencia SURF";
	string frameMaoBGR = "Referencia BGR";
	namedWindow(frameProfundidade, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMao, CV_WINDOW_AUTOSIZE);
    namedWindow(frameMaoBGR, CV_WINDOW_AUTOSIZE);
	namedWindow(frameSURF, CV_WINDOW_AUTOSIZE);

	//carrega o vocabulario em Bag-of-Words
    FileStorage fs("dicionario.xml", FileStorage::READ);
    fs["vocabulario"] >> vocabulario;
    fs.release();
	dextrator.setVocabulary(vocabulario);
	printf("\nVocabulario existente carregado!");

	//carrega o treinamento salvo pelo SVM para classificar os sinais
	//linearSVM.load("treinamento_1vs1_linear.xml");
	radialSVM.load("treinamento_1vs1_radial.xml");
	//polinomialSVM.load("treinamento_1vs1_polinomial.xml");
	printf("\nTreinamento existente carregado!\n\n");

	while (1) {
        sensor->waitForDeviceUpdateOnUser();

		//captura os dados do kinect para uso com o OpenCV
		VideoCapture capture(CV_CAP_OPENNI);
		capture.set(CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ);
		capture.grab();
		capture.retrieve(mapaBGR, CV_CAP_OPENNI_BGR_IMAGE);
		capture.retrieve(mapaProfundidade, CV_CAP_OPENNI_DISPARITY_MAP);

		if (sensor->getNumTrackedUsers() > 0) {
			//seta a frequencia
			QueryPerformanceFrequency(&frequencia);
			//inicia o timer pra contabilizar o tempo da extração de características
			QueryPerformanceCounter(&t1);
			//obtém o esqueleto
            Skeleton esqueleto =  sensor->getSkeleton(sensor->getUID(0));
            SkeletonPoint mao, cotovelo;
			//direita para o usuário
            mao = esqueleto.leftHand;
			cotovelo = esqueleto.leftElbow;
				
			//ajusta a região de interesse (mão) pela movimentação
			if (palma.x != 0) {
				if (!maoProximaPerimetro((float)palma.x, (float)palma.y)) {
					roi.x = palma.x - ROI_OFFSET;
					roi.y = palma.y - ROI_OFFSET;
				}//if (!maoProximaPerimetro(palma.x, palma.y))
			} else if (mao.confidence == 1.0) {
				if (!maoProximaPerimetro(mao.x, mao.y)) {
					roi.x = (int)mao.x - ROI_OFFSET;
					roi.y = (int)mao.y - ROI_OFFSET;
				}//if (!maoProximaPerimetro(mao.x, mao.y))
            }//if (mao.confidence == 1.0)
				
			//mapas coloridos da mão
			Mat mapaTempBGR(mapaBGR, roi);
			mapaMaoBGR = mapaTempBGR;
			//mapas de profundidade da mão
			Mat mapaTemp(mapaProfundidade, roi);
				
			//coloriza a disparidade encontrada pela câmera de profundidade
			Mat mapaDisparidadeColorido;
			colorizeDisparity(mapaTemp, mapaDisparidadeColorido, -1);
			mapaDisparidadeColorido.copyTo(mapaDisparidadeColorido, mapaTemp != 0);
				
			//aplica um blur pra tentar conter o ruído
			medianBlur(mapaDisparidadeColorido, mapaDisparidadeColorido, 5);

			//Descarta a profundidade a partir do OFFSET de cor
			//o elemento mais próximo (possívelmente a mão) e uma pequena distância a partir dele permancem
			int i, j, k = 0;
			for (i = 0; i < mapaDisparidadeColorido.rows; i++) {
				for (int j = 0; j < mapaDisparidadeColorido.cols; j++) {
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
				
			Mat mapaAux;
			vector<vector<Point>> contornos;
			//converte para cinza para realizar threshold e detectar a borda externa
			cvtColor(mapaDisparidadeColorido, mapaAux, CV_BGR2GRAY);
			//threshold de Otsu, adaptável, indetifica automático o nível adequado
			threshold(mapaAux, mapaAux, 50, 255, THRESH_BINARY+THRESH_OTSU);
			//busca os contornos
			findContours(mapaAux, contornos, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

			//se há contornos
			if (contornos.size()) {
				//busca o maior contorno (possívelmente a mão)
				int indiceMaior = 0;
				double area = 0, maiorArea = 0;
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
						mapaPontoInterno.at<float>(i,j) = (float)pointPolygonTest(contornos[indiceMaior], Point(i,j), true);
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
				int distanciaPonto = -1;
				for (i = 0; i < mapaPontoInterno.rows; i++) {
					for (j = 0; j < mapaPontoInterno.cols; j++) {
						if (mapaPontoInterno.at<float>(i,j) > distanciaPonto) {
							//evita encontrar círculos afastados do ponto médio, próximos da borda (antebraço
							if (pontoMedioY >= (j-abs(mapaPontoInterno.at<float>(i,j)))) {
								distanciaPonto = (int)mapaPontoInterno.at<float>(i,j);
								ix = i;
								jx = j;
							}//if (pontoMedioY >= (j-abs(mapaPontoInterno.at<float>(i,j))))
						}//if (mapaPontoInterno.at<float>(i,j) > distanciaPonto)
					}//for (j = 0; j < dist.cols; j++)
				}//for (i = 0; i < dist.rows; i++)
										
				//desenha o ponto e o maior círculo interno
				if (distanciaPonto > -1) {
					palma = Point(roi.x+ix, roi.y+jx);
					raioPalma = abs(distanciaPonto);
					//circle(mapaDisparidadeColorido, Point(ix, jx), 1, COR_AZUL, 1, CV_AA);
					//circle(mapaDisparidadeColorido, Point(ix, jx), abs(distanciaPonto), COR_AZUL, 1, CV_AA);
					
					//remove o espaço claros abaixo do círculo, possívelmente o antebraço, tenta poupar a mão se estiver abaixada (sinal m, n, q)
					for (i = (jx+distanciaPonto+5); i < mapaDisparidadeColorido.rows; i++) {
						for (j = 0; j < mapaDisparidadeColorido.cols; j++) {
							if (mapaDisparidadeColorido.at<Vec3b>(i, j)[R] != 0 && mapaDisparidadeColorido.at<Vec3b>(i, j)[G] > 40) {
								mapaDisparidadeColorido.at<Vec3b>(i, j)[B] = 0;
								mapaDisparidadeColorido.at<Vec3b>(i, j)[G] = 0;
								mapaDisparidadeColorido.at<Vec3b>(i, j)[R] = 0;
							}//if (mapaDisparidadeColorido.at<Vec3b>(i, j)[R] != 0)
						}//for ( j = 0; j < mapaDisparidadeColoridoValido.cols; j++)
					}//for( i = 0; i < mapaDisparidadeColoridoValido.rows; i++)	
				}//if (distanciaPonto > -1)
					
				//converte para cinza para realizar threshold e detectar as bordas com Canny
				cvtColor(mapaDisparidadeColorido, mapaAux, CV_BGR2GRAY);	
				//detectação de bordas Canny para ressaltar características internas da mão, threshold baixo (20 ~ 40) para detectar mais características
				Canny(mapaAux, mapaAux, 20, 20*2);	
				//busca as bordas de Canny para desenhar
				findContours(mapaAux, contornos, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
				for (i = 0; i < contornos.size(); i++) {
					drawContours(mapaDisparidadeColorido, contornos, i, COR_BRANCO, 1, 8, noArray(), 0, Point() );
				}//for (i = 0; i < contornosCanny.size(); i++)
			}//if (contornos.size()) {

			//transforma o mapa em cinza para o SURF
			cvtColor(mapaDisparidadeColorido, mapaSURF, CV_BGR2GRAY);
				
			//extrai os descritores com o SURF considerando o vocabulário da Bag-of-Words 
			Mat descritor;
			vector<KeyPoint> pontosChave;
			detetor->detect(mapaSURF, pontosChave);
			dextrator.compute(mapaSURF, pontosChave, descritor);

			//pega o timer do fim da extração e computa o tempo decorrido
			QueryPerformanceCounter(&t2);
			//usa o svm e imprime o resultado
			if (TREINAMENTO == 0 && descritor.rows > 0) {
				if (quantidadeReconhecimentoEtapa < REPETICAOETAPA) {
					//mede o tempo decorrido para extração de características
					dados[quantidadeReconhecimento][TEMPO_EXTRACAO] = medirTempoDecorrido(t1, t2);
					printf("\n\nEXTRACAO DE CARACTERISTICAS %fms\nRECONHECIMENTO:", dados[quantidadeReconhecimento][TEMPO_EXTRACAO]);

					//predição linear, contabiliza o tempo
					printf("\nSVM (1vs1) LINEAR: ");
					QueryPerformanceCounter(&t3);
					dados[quantidadeReconhecimento][RESPOSTA] = linearSVM.predict(descritor);
					QueryPerformanceCounter(&t4);
					dados[quantidadeReconhecimento][TEMPO] = medirTempoDecorrido(t3, t4);
					imprimirResposta(dados[quantidadeReconhecimento][RESPOSTA], dados[quantidadeReconhecimento][TEMPO]);
					//predição polinomial, contabiliza o tempo
					/*printf("\nSVM (1vs1) POLINOMIAL: ");
					QueryPerformanceCounter(&t5);
					dados[quantidadeReconhecimento][RESPOSTA] = polinomialSVM.predict(descritor);
					QueryPerformanceCounter(&t6);
					dados[quantidadeReconhecimento][TEMPO] = medirTempoDecorrido(t5, t6);
					imprimirResposta(dados[quantidadeReconhecimento][RESPOSTA], dados[quantidadeReconhecimento][TEMPO]);
					//predição radial, contabiliza o tempo
					/*printf("\nSVM (1vs1) RADIAL: ");
					QueryPerformanceCounter(&t7);
					dados[quantidadeReconhecimento][RESPOSTA] = radialSVM.predict(descritor);
					QueryPerformanceCounter(&t8);
					dados[quantidadeReconhecimento][TEMPO] = medirTempoDecorrido(t7, t8);
					imprimirResposta(dados[quantidadeReconhecimento][RESPOSTA], dados[quantidadeReconhecimento][TEMPO]);*/
						
					quantidadeReconhecimento++;
					quantidadeReconhecimentoEtapa++;
				} else {

					printf("\n Completou uma etapa de %d reconhecimentos!", quantidadeReconhecimentoEtapa);
					quantidadeReconhecimentoEtapa = 0;
					TREINAMENTO = 1;

				}//if (quantidadeReconhecimento < REPETICAOETAPA)
			}//if (TREINAMENTO == 0)*/

			mapaMao = mapaDisparidadeColorido;
			
			//só por debugar
			drawKeypoints(mapaSURF, pontosChave, mapaSURF, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

			//desenha o esqueleto do torso, braços e cabeça no mapa de profundidade colorido, apenas para referência, não contabiliza o tempo
			//cria o mapa de profundidade colorido, para observar depuração dos esqueleto e da ROI
			cvtColor(mapaProfundidade, mapaProfundidadeEsqueleto, CV_GRAY2BGR);
			//recebe os pontos pelo kinect
			Point torso((int)esqueleto.torso.x, (int)esqueleto.torso.y);
			Point pescoco((int)esqueleto.neck.x, (int)esqueleto.neck.y);
			Point cabeca((int)esqueleto.head.x, (int)esqueleto.head.y);
			Point maoEsq = Point((int)esqueleto.leftHand.x, (int)esqueleto.leftHand.y);
			Point cotoveloEsq = Point((int)esqueleto.leftElbow.x, (int)esqueleto.leftElbow.y);
			Point ombroEsq = Point((int)esqueleto.leftShoulder.x, (int)esqueleto.leftShoulder.y);
			Point maoDir = Point((int)esqueleto.rightHand.x, (int)esqueleto.rightHand.y);
			Point cotoveloDir = Point((int)esqueleto.rightElbow.x, (int)esqueleto.rightElbow.y);
			Point ombroDir = Point((int)esqueleto.rightShoulder.x, (int)esqueleto.rightShoulder.y);
			//desenha a estrutura do braço esquerdo
			circle(mapaProfundidadeEsqueleto, maoEsq, 2, COR_AMARELO, 4);
			circle(mapaProfundidadeEsqueleto, cotoveloEsq, 2, COR_AMARELO, 4);
			circle(mapaProfundidadeEsqueleto, ombroEsq, 2, COR_AMARELO, 4);
			line(mapaProfundidadeEsqueleto, maoEsq, cotoveloEsq, COR_AMARELO, 2);
			line(mapaProfundidadeEsqueleto, cotoveloEsq, ombroEsq, COR_AMARELO, 2);
			//desenha a estrutura do braço direito
			circle(mapaProfundidadeEsqueleto, maoDir, 2, COR_VERDE, 4);
			circle(mapaProfundidadeEsqueleto, cotoveloDir, 2, COR_VERDE, 4);
			circle(mapaProfundidadeEsqueleto, ombroDir, 2, COR_VERDE, 4);
			line(mapaProfundidadeEsqueleto, maoDir, cotoveloDir, COR_VERDE, 2);
			line(mapaProfundidadeEsqueleto, cotoveloDir, ombroDir, COR_VERDE, 2);
			//desenha a estrutura do torso e cabeça
			circle(mapaProfundidadeEsqueleto, torso, 2, COR_BRANCO, 4);
			circle(mapaProfundidadeEsqueleto, pescoco, 2, COR_AZUL, 4);
			line(mapaProfundidadeEsqueleto, ombroEsq, torso, COR_BRANCO, 2);
			line(mapaProfundidadeEsqueleto, ombroEsq, pescoco, COR_BRANCO, 2);
			line(mapaProfundidadeEsqueleto, ombroDir, torso, COR_BRANCO, 2);
			line(mapaProfundidadeEsqueleto, ombroDir, pescoco, COR_BRANCO, 2);
			circle(mapaProfundidadeEsqueleto, cabeca, 2, COR_AZUL, 4);
			line(mapaProfundidadeEsqueleto, pescoco, cabeca, COR_AZUL, 2);
			//desenha o círculo interno (possívelmente a palma) no mapa do esqueleto
			circle(mapaProfundidadeEsqueleto, palma, 1, COR_VERMELHO, 2);
			circle(mapaProfundidadeEsqueleto, palma, raioPalma, COR_VERMELHO, 2);
			//desenha o retângulo da região de interesse para referência
			rectangle(mapaProfundidadeEsqueleto, Point(roi.x, roi.y), Point(roi.x+ROIXRES, roi.y+ROIYRES), COR_BRANCO, 2);
		}//if (sensor->getNumTrackedUsers() > 0)

		//recebe a interação do teclado
		int tecla = waitKey(10);
		if (tecla > 0) {
			teclado = tecla;
		}//if (tecla > 0)
		
		int sinal = (int)retornarSinalPorTecla(teclado);
		if (sinal != 0) {
			if (quantidadeTreinamento[sinal] >= REPETICAO) {
				printf("\nLimite de captura letra %c!", teclado);
				teclado = 999;
			} else {
				armazenador.push_back(mapaSURF.clone());
				quantidadeTreinamento[sinal]++;
				//mede o tempo decorrido para extração de características
				dados[quantidadeReconhecimento][TEMPO_EXTRACAO] = medirTempoDecorrido(t1, t2);
				quantidadeReconhecimento++;
				printf("\nFrame capturado da letra %c!", teclado);
				if (quantidadeTreinamento[sinal] == REPETICAOETAPA) {
					printf("\nEtapa concluida letra %c!", teclado);
					teclado = 999;
				}//if (quantidadeTreinamento[sinal] == REPETICAOETAPA)
			}//if (quantidadeTreinamento[sinal] >= REPETICAO)

		} else if (teclado == ' ') {

			if (TREINAMENTO == 1) {
				printf("\nIniciando reconhecimento!");
				TREINAMENTO = 0;
				teclado = 999;
			}//if (TREINAMENTO == 0)

		} else if (teclado == '1') {

			if (quantidadeVocabulario < REPETICAO) {
				Mat descritoresVocabulario;
				vector<KeyPoint> pontosChaveVocabulario;
				armazenador.push_back(mapaSURF.clone());
				detetor->detect(mapaSURF, pontosChaveVocabulario);
				extrator->compute(mapaSURF, pontosChaveVocabulario, descritoresVocabulario);
				caracteristicasDesagrupadas.push_back(descritoresVocabulario);
				printf("\nFrame capturado para vocabulario! Descritores SURF capturados! (%d)", descritoresVocabulario.size().height);
				quantidadeVocabulario++;
			} else {
				printf("\nEtapa de captura de vocabulario concluida!");
				quantidadeVocabulario = 0;
				teclado = 999;
			}//if (quantidadeVocabulario < REPETICAO)

		} else if (teclado == '2') {

			for (int i = 0; i < armazenador.size(); i++) {
				stringstream ss;
				string name = "vocabulario_";
				string type = ".png";
				ss << name << (i+1) << type;
				string file = ss.str();
				imwrite(file, armazenador[i]);
			}//for (int i = 0; i < armazenador.size(); i++) 

			BOWKMeansTrainer bagOfWords(caracteristicasDesagrupadas.size().height, TermCriteria(CV_TERMCRIT_ITER,100,0.0001), 1, KMEANS_PP_CENTERS);			
			vocabulario = bagOfWords.cluster(caracteristicasDesagrupadas);

			FileStorage fs("dicionario.xml", FileStorage::WRITE);
			fs << "vocabulario" << vocabulario;
			fs.release();
			printf("\nVocabulario salvo!");

			teclado = 999;
			break;

		} else if (teclado == '3') {

			for (int i = 0; i < armazenador.size(); i++) {
				stringstream ss;
				string name = "treinamento_";
				string type = ".png";
				ss << name << (i+1) << type;
				string file = ss.str();
				imwrite(file, armazenador[i]);
			}//for (int i = 0; i < armazenador.size(); i++)
			teclado = 999;

		} else if (teclado == '4') {

			treinarEtapa("08", dadosTreinamento, labelTreinamento, detetor, dextrator);
			treinarEtapa("10", dadosTreinamento, labelTreinamento, detetor, dextrator);
			treinarEtapa("12", dadosTreinamento, labelTreinamento, detetor, dextrator);
			treinarEtapa("14", dadosTreinamento, labelTreinamento, detetor, dextrator);
			treinarEtapa("16", dadosTreinamento, labelTreinamento, detetor, dextrator);

			printf("\n Treinamento: Dados %d - Labels %d", dadosTreinamento.rows, labelTreinamento.rows);
			//salva o treinamento SVM
			//treinamento linear
			linearSVMparams.C = 3.1250000000000000e+002;
			CvParamGrid gradeC1(linearSVMparams.C, linearSVMparams.C, 0);
			//linearSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), linearSVMparams);
			linearSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), linearSVMparams
				, 10, gradeC1, CvSVM::get_default_grid(CvSVM::GAMMA), CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), false);
			linearSVM.save("treinamento_1vs1_linear.xml");
			printf("\n Treinamento linear encerrado");


			//treinamento radial
			radialSVMparams.gamma = 5.0625000000000009e-001;
			radialSVMparams.C = 3.1250000000000000e+002;
			CvParamGrid gradeGamma2(radialSVMparams.gamma, radialSVMparams.gamma, 0);
			CvParamGrid gradeC2(radialSVMparams.C, radialSVMparams.C, 0);
			//radialSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), radialSVMparams);
			radialSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), radialSVMparams
				, 10, gradeC2, gradeGamma2, CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), false);
			radialSVM.save("treinamento_1vs1_radial.xml");
			printf("\n Treinamento radial encerrado");
			
			
			//treinamento polinomial
			polinomialSVMparams.degree = 7.0000000000000007e-002;
			polinomialSVMparams.gamma = 5.0625000000000009e-001;
			polinomialSVMparams.coef0 = 1.0000000000000001e-001;
			polinomialSVMparams.C = 3.1250000000000000e+002;
			CvParamGrid gradeDegree3(polinomialSVMparams.degree, polinomialSVMparams.degree, 0);
			CvParamGrid gradeGamma3(polinomialSVMparams.gamma, polinomialSVMparams.gamma, 0);
			CvParamGrid gradeCoef3(polinomialSVMparams.coef0, polinomialSVMparams.coef0, 0);
			CvParamGrid gradeC3(polinomialSVMparams.C, polinomialSVMparams.C, 0);
			//polinomialSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), polinomialSVMparams);
			polinomialSVM.train_auto(dadosTreinamento, labelTreinamento, Mat(), Mat(), polinomialSVMparams
				, 10, gradeC3, gradeGamma3, CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), gradeCoef3, gradeDegree3, false);
			polinomialSVM.save("treinamento_1vs1_polinomial.xml");
			printf("\n Treinamento polinomial encerrado");


			teclado = 999;
			break;
		
		} else if (teclado == '5') {

			printf("\n Validacao de treinamento iniciada!");
			//validarEtapa("treinamento08m/treinamento_", linearSVM, detetor, dextrator);
			//validarEtapa("treinamento10m/treinamento_", linearSVM, detetor, dextrator);
			//validarEtapa("treinamento12m/treinamento_", linearSVM, detetor, dextrator);
			//validarEtapa("treinamento08m/treinamento_", radialSVM, detetor, dextrator);
			//validarEtapa("treinamento10m/treinamento_", radialSVM, detetor, dextrator);
			//validarEtapa("treinamento12m/treinamento_", radialSVM, detetor, dextrator);
			validarEtapa("treinamento08m/treinamento_", polinomialSVM, detetor, dextrator);
			validarEtapa("treinamento10m/treinamento_", polinomialSVM, detetor, dextrator);
			validarEtapa("treinamento12m/treinamento_", polinomialSVM, detetor, dextrator);
			printf("\n Validacao de treinamento concluida!");
			teclado = 999;

		} else if (teclado == '6') {

			printf("\n Validacao de iniciada!");
			//validarEtapa("validacao08m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao10m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao12m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao13m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao14m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao15m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao16m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao18m/validacao_", linearSVM, detetor, dextrator);
			//validarEtapa("validacao20m/validacao_", linearSVM, detetor, dextrator);

			//validarEtapa("validacao08m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao10m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao12m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao13m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao14m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao15m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao16m/validacao_", radialSVM, detetor, dextrator);
			//validarEtapa("validacao18m/validacao_", radialSVM, detetor, dextrator);
			validarEtapa("validacao20m/validacao_", radialSVM, detetor, dextrator);

			//validarEtapa("validacao08m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao10m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao12m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao13m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao14m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao15m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao16m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao18m/validacao_", polinomialSVM, detetor, dextrator);
			//validarEtapa("validacao20m/validacao_", polinomialSVM, detetor, dextrator);

			printf("\n Validacao de concluida!");
			teclado = 999;

		} else if (teclado == '7') {

			for (int i = 0; i < armazenador.size(); i++) {
				stringstream ss;
				string name = "validacao_";
				string type = ".png";
				ss << name << (i+1) << type;
				string file = ss.str();
				imwrite(file, armazenador[i]);
			}//for (int i = 0; i < armazenador.size(); i++)
			ofstream fs("resultado.csv");
			for (int i = 0; i < armazenador.size(); i++) {
				fs << dados[i][TEMPO_EXTRACAO]
				<< endl;
			}//for (int i = 0; i < armazenador.size(); i++)
			teclado = 999;

		} else if (teclado == 27) {
			ofstream fs("resultado.csv");
			for (int i = 0; i < (REPETICAOETAPA*20); i++) {				
				fs << dados[i][TEMPO_EXTRACAO]
				<< ";" << letra[(int)dados[i][RESPOSTA]] << ";" << dados[i][TEMPO]
				<< endl;
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

		//apresenta a imagem da mão
		if (mapaMao.rows > 0) {
			resize(mapaMao, mapaMao, Size(), 3, 3);
			resize(mapaMaoBGR, mapaMaoBGR, Size(), 2, 2);
			resize(mapaSURF, mapaSURF, Size(), 2, 2);
			imshow(frameMao, mapaMao);
			imshow(frameMaoBGR, mapaMaoBGR);
			imshow(frameSURF, mapaSURF);
        }//if (mapaMao.rows > 0)

		mapaMao.setTo(0);
		mapaSURF.setTo(0);
    }//while (1)

	delete sensor;

    return 0;
}