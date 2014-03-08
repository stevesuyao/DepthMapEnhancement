#ifndef GLCAPTURE_H
#define GLCAPTURE_H
#include <GL/glut.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <strstream>

//初期のコーデック、パソコンに入ってる好きなコーデックにする
#define DEFAULT_CODEC CV_FOURCC('X','V','I','D')

/*動画コーデック例*/
//#define DEFAULT_CODEC CV_FOURCC('P','I','M','1')	//MPEG-1
//#define DEFAULT_CODEC CV_FOURCC('M','P','G','4')	//MPEG-4
//#define DEFAULT_CODEC CV_FOURCC('M','P','4','2')	//MPEG-4.2
//#define DEFAULT_CODEC CV_FOURCC('D','I','V','X')	//DivX
//#define DEFAULT_CODEC CV_FOURCC('D','X','5','0')	//DivX ver 5.0
//#define DEFAULT_CODEC CV_FOURCC('X','V','I','D')	//Xvid
//#define DEFAULT_CODEC CV_FOURCC('U','2','6','3')	//H.263
//#define DEFAULT_CODEC CV_FOURCC('H','2','6','4')	//H.264
//#define DEFAULT_CODEC CV_FOURCC('F','L','V','1')	//FLV1
//#define DEFAULT_CODEC CV_FOURCC('M','J','P','G')	//Motion JPEG
//#define DEFAULT_CODEC 0							//非圧縮
//#define DEFAULT_CODEC -1							//選択


//GL_TEXTURE_2Dを使った方法（こっちのほうが速いという噂）
#define USE_TEXTURE_2D

class GLCapture{
private:
	cv::Mat captureImage;
	cv::VideoWriter videoWriter;
	int interpolation;	//補間手法
	int count;
	const char* name;
public:
	GLCapture():count(0){
		interpolation = cv::INTER_LINEAR;
	}
	/**
	const char* filename	ファイル名
	int fourcc				コーデック
	double fps				fps
	cv::Size size			サイズ
	*/
	GLCapture(const char* filename,int fourcc = DEFAULT_CODEC,double fps = 3.0, cv::Size size = cv::Size(glutGet(GLUT_WINDOW_WIDTH),glutGet( GLUT_WINDOW_HEIGHT))):
		count(0){
		name = filename;
		std::ostringstream videoname;
		videoname << filename << ".avi";
		setWriteFile((videoname.str()).c_str(),fourcc, fps, size);
	}
	~GLCapture(){
		videoWriter.release();
	}
	/**
	const char* filename	ファイル名
	int fourcc				コーデック
	double fps				fps
	cv::Size size			サイズ
	*/
	void setWriteFile(const char* filename = "output",int fourcc = DEFAULT_CODEC,double fps = 3.0, cv::Size size = cv::Size(glutGet(GLUT_WINDOW_WIDTH),glutGet( GLUT_WINDOW_HEIGHT))){
		name = filename;
		std::ostringstream videoname;
		videoname << filename << ".avi";
		videoWriter = cv::VideoWriter::VideoWriter((videoname.str()).c_str(),fourcc, fps, size); 
		captureImage = cv::Mat(size,CV_8UC3);
	}
	/**
		動画ファイルに書き込み
	*/
	bool write(){
		if( videoWriter.isOpened()){
			cv::Size size = cv::Size(glutGet(GLUT_WINDOW_WIDTH),glutGet( GLUT_WINDOW_HEIGHT));
			if(size.width == captureImage.cols && size.height == captureImage.rows){
#ifndef USE_TEXTURE_2D
				glReadPixels(0,0, captureImage.cols,captureImage.rows,GL_RGB,GL_UNSIGNED_BYTE,captureImage.data);
#else
				glCopyTexImage2D( GL_TEXTURE_2D,0,GL_RGB,0,0,captureImage.cols,captureImage.rows,0);
				glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE,captureImage.data);
#endif
			}
			else{	//ウィンドウが拡大縮小されている場合
				size.width -= size.width % 4;	//4の倍数にする
				cv::Mat temp(size,CV_8UC3);
#ifndef USE_TEXTURE_2D
				glReadPixels(0,0, size.width,size.height,GL_RGB,GL_UNSIGNED_BYTE,temp.data);
#else
				glCopyTexImage2D( GL_TEXTURE_2D,0,GL_RGB,0,0,size.width,size.height,0);
				glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE,temp.data);
#endif
				cv::resize(temp,captureImage,captureImage.size(),0.0,0.0,interpolation);
			}
			cvtColor(captureImage,captureImage,CV_RGB2BGR);   
			flip(captureImage,captureImage,0);
			videoWriter << captureImage;
			//char key = cv::waitKey(1);
			//std::ostringstream oss;
			//oss << name << count << ".bmp";
			//if(key == 's'){
			//	cv::imwrite(oss.str(), captureImage);
			//	count++;
			//}
			return true;

		}
		else
			return false;
	}
	/**
		拡大縮小時の補間方法を取得
	*/
	int getInterpolation(){
		return interpolation;
	}
	/**
		拡大縮小時の補間方法を設定
	*/
	void setInterpolation(int interpolation){
		this->interpolation = interpolation;
	}

};
#endif