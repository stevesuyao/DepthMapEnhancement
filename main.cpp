#include "Kinect.h"
#include "Segmentation_SLIC.h"
#include "SuperpixelSegmentation.h"
#include "DepthAdaptiveSuperpixel.h"
#include "JointBilateralFilter.h"
#include "DimensionConvertor.h"
#include <time.h>
#include "NormalEstimation\NormalMapGenerator.h"
#include "NormalAdaptiveSuperpixel.h"
#include "ArrayBuffer\Buffer2D.h"
#include "LabelEquivalenceSeg.h"
#include <opencv2\opencv.hpp>
#include "VisualizeSurfaceMesh.h"
#include <opencv2\gpu\gpu.hpp>
#include "Projection_GPU.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "EdgeRefinedSuperpixel.h"
#include "MarkovRandomField.h"

float color_sigma = 250.0f;
float spatial_sigma = 15.0f;
float depth_sigma = 20.0f;
float normal_sigma = 80.0f;
int iteration = 5;
const int sp_rows = 10;
const int sp_cols = 20;

void colorChange(int change, void* dummy){
	color_sigma = change;
}
void spatialChange(int change, void* dummy){
	spatial_sigma = change;
}
void depthChange(int change, void* dummy){
	depth_sigma = change;
}
void normalChange(int change, void* dummy){
	normal_sigma = change;
}

//int main(int argc, char** argv) {
//	//cv::gpu::DeviceInfo di(cv::gpu::getDevice());
//	//std::cout << di.name() << std::endl;
//
//	try{
//
//		/*cv::Mat input = cv::imread("image.jpg", 1);
//		cv::gpu::GpuMat input_gpu(input.rows, input.cols, CV_8UC3);
//		input_gpu.upload(input);
//		cv::gpu::GpuMat resized_gpu(input.rows*2, input.cols*2, CV_8UC3);
//		cv::gpu::resize(input_gpu, resized_gpu, cv::Size(input.cols*2, input.rows*2), cv::INTER_LINEAR);
//		cv::Mat resized(input.rows*2, input.cols*2, CV_8UC3);
//		resized_gpu.download(resized);
//		cv::imshow("resized", resized);
//		cv::waitKey();
//		*/
//		///////////////////////////
//		VisualizeSurfaceMesh Visualize;
//		Visualize.Process(argc, argv);
//	}
//	catch (std::exception& ex) {
//		std::cout << ex.what() << std::endl;
//		char end_key;
//		for(;;)
//			end_key = cv::waitKey(10);
//	}
//	return 0;
//}


cv::VideoWriter	segment_sp_writer;
cv::VideoWriter	random_sp_writer;

cv::VideoWriter	segment_ndsp_writer;
cv::VideoWriter	random_ndsp_writer;

cv::VideoWriter normal_writer;
cv::VideoWriter seg_normal_writer;



void checkKey(NormalMapGenerator& nmg, char key){
	switch(key){
	case '\033':
		segment_sp_writer.release();
		random_sp_writer.release();
		segment_ndsp_writer.release();
		random_ndsp_writer.release();
		normal_writer.release();
		seg_normal_writer.release();
		exit(1);
		break;
	case 'q':
		exit(0);
		break;
	case '1':
		nmg.setNormalEstimationMethods(nmg.BILATERAL);
		printf("NormalEstimation method: BILATERAL\n");
		break;
	case '2':
		nmg.setNormalEstimationMethods(nmg.SDC);
		printf("NormalEstimation method: SDC\n");
		break;
	case '3':
		nmg.setNormalEstimationMethods(nmg.CM);
		printf("NormalEstimation method: CM\n");
		break;

	default:
		break;
	}
	return;
}

int main(){
	segment_sp_writer = cv::VideoWriter::VideoWriter("segment_sp.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(Kinect::Width, Kinect::Height));
	random_sp_writer = cv::VideoWriter::VideoWriter("random_sp.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(Kinect::Width, Kinect::Height));
	segment_ndsp_writer = cv::VideoWriter::VideoWriter("segment_ndsp.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(Kinect::Width, Kinect::Height));
	random_ndsp_writer = cv::VideoWriter::VideoWriter("random_ndsp.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(Kinect::Width, Kinect::Height));
	normal_writer = cv::VideoWriter::VideoWriter("normal.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(Kinect::Width, Kinect::Height));
	seg_normal_writer = cv::VideoWriter::VideoWriter("seg_nromal.avi", CV_FOURCC('X','V','I','D'), 3.0, cv::Size(Kinect::Width, Kinect::Height));

	SingleKinect kinect;
	Segmentation_SLIC SLIC(Kinect::Width, Kinect::Height);
	SLIC.SetParametor(240, 30.0, 10);
	cv::Mat_<cv::Vec3b> lab = cv::imread("lab.jpg", 1);
	cv::Mat_<cv::Vec3b> lab2;
	lab.copyTo(lab2);
	for(int y=0; y<lab.rows; y++){
		for(int x=0; x<lab.cols; x++){
			lab2.at<cv::Vec3b>(y, lab.cols-x) = lab.at<cv::Vec3b>(y, x);
		}
	}
	//cv::imwrite("lab_inv.bmp", lab2);
	//SLIC.segmentImage_SLIC(lab2);
	//SLIC.VisualizeSegmentedImage();
	//cv::imwrite("lab_segment.bmp", SLIC.getRandomImg());
	SuperpixelSegmentation Seg(Kinect::Width, Kinect::Height);
	Seg.SetParametor(sp_rows, sp_cols);
	DepthAdaptiveSuperpixel DASP(Kinect::Width, Kinect::Height);
	DASP.SetParametor(sp_rows, sp_cols, kinect.GetIntrinsicMatrix());
	NormalAdaptiveSuperpixel NASP(Kinect::Width, Kinect::Height);
	NASP.SetParametor(sp_rows, sp_cols, kinect.GetIntrinsicMatrix());
	//Visualizer visual(Kinect::Width, Kinect::Height);
	cv::Mat_<int> label(Kinect::Height, Kinect::Width);

	clock_t start;
	double prev, mine, jbf;
	float* inputDepth_Host, *inputDepth_Device, *bufferDepth_Host, *bufferDepth_Device;
	float3* points_Host, *points_Device, *refinedPoints_Device, *refinedPoints_Host;
	cudaMallocHost(&bufferDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&bufferDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&inputDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMalloc(&inputDepth_Device, sizeof(float)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&points_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&points_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&points_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&points_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMallocHost(&refinedPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
	cudaMalloc(&refinedPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height);
	
	//array buffer
	Buffer2D Buffer(Kinect::Width, Kinect::Height);
	//joint bilateral filter
	JointBilateralFilter JBF(Kinect::Width, Kinect::Height);
	cv::gpu::GpuMat Color_Device = cv::gpu::createContinuous(Kinect::Height, Kinect::Width, CV_8UC3);
	//cv::gpu::GpuMat Color_Device(Kinect::Height, Kinect::Width, CV_8UC3);
	//dimension convertor
	DimensionConvertor convertor;
	convertor.setCameraParameters(kinect.GetIntrinsicMatrix(), Kinect::Width, Kinect::Height);
	//normal generator
	NormalMapGenerator nmg(Kinect::Width, Kinect::Height);
	nmg.setNormalEstimationMethods(nmg.BILATERAL);
	//labeling
	LabelEquivalenceSeg spMerger(Kinect::Width, Kinect::Height);
	//projection
	Projection_GPU Projector(Kinect::Width, Kinect::Height, kinect.GetIntrinsicMatrix());
	//visualize
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr input (new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr upsampled (new pcl::PointCloud<pcl::PointXYZRGB>);
	//edge refine
	EdgeRefinedSuperpixel ERS(Kinect::Width, Kinect::Height);
	//markov random field
	MarkovRandomField MRF(Kinect::Width, Kinect::Height);
	cv::Mat_<cv::Vec3b> normalImg(Kinect::Height, Kinect::Width);
	while(true){
		kinect.UpdateContextAndData();
		//start = clock();	
		//label = SLIC.segmentImage_SLIC(kinect.GetColorImage());
		//prev = (double)((clock() - start)/*/1000.0*/);
		//std::cout << "prev: "<<prev<<std::endl;
		//SLIC.VisualizeSegmentedImage();
		//visual.display(kinect.GetColorImage(), label, "seg");
		
		Color_Device.upload(kinect.GetColorImage());
			//depth input
		for(int y=0; y<Kinect::Height; y++){
			for(int x=0; x<Kinect::Width; x++){
				inputDepth_Host[y*Kinect::Width+x] = (float)(*kinect.GetDepthMD())(x, y);
			}
		}
		cudaMemcpy(inputDepth_Device, inputDepth_Host, sizeof(float)*Kinect::Width*Kinect::Height, cudaMemcpyHostToDevice);
		//joint bilateral filter
		JBF.Process(inputDepth_Device, Color_Device);
		//markov random field
		MRF.Process(inputDepth_Device, Color_Device);
		//convert to realworld
		convertor.projectiveToReal(JBF.getFiltered_Device(), points_Device);
		//Superpixel Segmentation	
		Seg.Process(JBF.getSmoothImage_Device(), 180.0f, 3.0f, 5);
		//Depth Adaptive Superpixels
		DASP.Segmentation(JBF.getSmoothImage_Device(), points_Device, 0.0f, 10.0f, 200.0f, 5);
		//cv::imshow("DASP", DASP.getSegmentedImage(kinect.GetDepthImage(), NASP.Line));
		//cv::imshow("SP", Seg.getSegmentedImage(kinect.GetColorImage(), NASP.Line));
		//cv::imshow("depth", kinect.GetDepthImage());
		
		//ERS.EdgeRefining(Seg.getLabelDevice(), DASP.getLabelDevice(), inputDepth_Device, Color_Device);
		//cv::imshow("Refine", ERS.getSegmentedImage(kinect.getMaxDepth()));
		//MRF.visualize(inputDepth_Host);
		//convert to realworld
		//convertor.projectiveToReal(ERS.getRefinedDepth_Device(), refinedPoints_Device);
		convertor.projectiveToReal(JBF.getFiltered_Device(), refinedPoints_Device);
		//cudaMemcpy(refinedPoints_Host, refinedPoints_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
		//cudaMemcpy(points_Host, points_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
		//for(int y=0; y<Kinect::Height; y++){
		//	for(int x=0; x<Kinect::Width; x++){
		//		std::cout << "points_Host "<< points_Host[y*Kinect::Width+x].x<<", "<<points_Host[y*Kinect::Width+x].y<<", "<<points_Host[y*Kinect::Width+x].z<<std::endl;
		//		std::cout << "refinedPoints_Host "<< refinedPoints_Host[y*Kinect::Width+x].x<<", "<<refinedPoints_Host[y*Kinect::Width+x].y<<", "<<refinedPoints_Host[y*Kinect::Width+x].z<<std::endl;
		//	}
		//}
		//Normal estimation
		nmg.generateNormalMap(points_Device);
		NASP.Segmentation(JBF.getSmoothImage_Device(), points_Device, nmg.getNormalMap(), color_sigma, spatial_sigma, depth_sigma, normal_sigma, iteration);
		////refine edge
		ERS.EdgeRefining(Seg.getLabelDevice(), NASP.getLabelDevice(), JBF.getFiltered_Device(), Color_Device);
		cv::imshow("Refine", ERS.getSegmentedImage(kinect.getMaxDepth()));
		convertor.projectiveToReal(ERS.getRefinedDepth_Device(), refinedPoints_Device);
		for(int y=0; y<Kinect::Height; y++){
			for(int x=0; x<Kinect::Width; x++){
				int label = ERS.getRefinedLabels_Host()[y*Kinect::Width+x];
				if(label != -1){
					//std::cout << NASP.getNormalsHost()[label].x << ", "<<NASP.getNormalsHost()[label].y<<", "<<NASP.getNormalsHost()[label].z<<std::endl;
					normalImg.at<cv::Vec3b>(y, x).val[0]= (unsigned char)(255*(NASP.getNormalsHost()[label].x+1.0)/2);
					normalImg.at<cv::Vec3b>(y, x).val[1]= (unsigned char)(255*(NASP.getNormalsHost()[label].y+1.0)/2);
					normalImg.at<cv::Vec3b>(y, x).val[2]= (unsigned char)(255*(NASP.getNormalsHost()[label].z+1.0)/2);
				}
				else
					normalImg.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}
		}
		cv::imshow("segmented image", NASP.getSegmentedImage(kinect.GetColorImage(), NASP.Line));
		segment_ndsp_writer << NASP.getSegmentedImage(kinect.GetColorImage(), NASP.Line);
		cv::imshow("random image", NASP.getRandomColorImage());
		random_ndsp_writer <<  NASP.getRandomColorImage();
		//cv::imshow("cluster normal", NASP.getNormalImg());
		//cv::imshow("cluster normal", normalImg);
		//seg_normal_writer << NASP.getNormalImg();
		cv::imshow("normal", nmg.getNormalImg());
		normal_writer <<nmg.getNormalImg();
		//labeling
		spMerger.labelImage(NASP.getNormalsDevice(), NASP.getLabelDevice(), NASP.getCentersDevice(), NASP.getNormalsVarianceDevice(), (Kinect::Width/sp_rows)*(Kinect::Height/sp_cols));
		spMerger.viewSegmentResult();
		cv::imshow("normal_label", spMerger.getNormalImg());
		char key = cv::waitKey(1);
		checkKey(nmg, key);
		if(key == 'c'){
			if(color_sigma >=5.0f)
			color_sigma -= 5.0f;
			std::cout << "color_sigma: "<<color_sigma << std::endl; 
		}
		if(key == 'C'){
			color_sigma += 5.0f;
			std::cout << "color_sigma: "<<color_sigma << std::endl; 
		}
		if(key == 's'){
			if(spatial_sigma >=5.0f)
			spatial_sigma -= 5.0f;
			std::cout << "spatial_sigma: "<<spatial_sigma << std::endl; 
		}
		if(key == 'S'){
			spatial_sigma += 5.0f;
			std::cout << "spatial_sigma: "<<spatial_sigma << std::endl; 
		}
		if(key == 'd'){
			if(depth_sigma >=5.0f)
			depth_sigma -= 5.0f;
			std::cout << "depth_sigma: "<<depth_sigma << std::endl; 
		}
		if(key == 'D'){
			depth_sigma += 5.0f;
			std::cout << "depth_sigma: "<<depth_sigma << std::endl; 
		}
		if(key == 'n'){
			if(normal_sigma >=5.0f)
			normal_sigma -= 5.0f;
			std::cout << "normal_sigma: "<<normal_sigma << std::endl; 
		}
		if(key == 'N'){
			normal_sigma += 5.0f;
			std::cout << "normal_sigma: "<<normal_sigma << std::endl; 
		}
		if(key == 'i'){
			if(iteration >= 2)
			iteration--;
			std::cout << "iteration: "<<iteration << std::endl; 
		}
		if(key == 'I'){
			iteration++;
			std::cout << "iteration: "<<iteration << std::endl; 
		}
			//GUI
		cv::createTrackbar("color_sigma", "random image", NULL, 400, colorChange);
		cv::setTrackbarPos("color_sigma", "random image", color_sigma);
		cv::createTrackbar("spatial_sigma", "random image", NULL, 100, spatialChange);
		cv::setTrackbarPos("spatial_sigma", "random image", spatial_sigma);
		cv::createTrackbar("depth_sigma", "random image", NULL, 200, depthChange);
		cv::setTrackbarPos("depth_sigma", "random image", depth_sigma);
		cv::createTrackbar("normal_sigma", "random image", NULL, 200, normalChange);
		cv::setTrackbarPos("normal_sigma", "random image", normal_sigma);

		if(key == 'p'){
		//for(int y=0; y<Kinect::Height; y++){
		//		for(int x=0; x<Kinect::Width; x++){
		//	std::cout << "label: "<<spMerger.getMergedClusterLabel_Host()[y*Kinect::Width+x] <<std::endl;
		//		std::cout << "x, y, z, w "<<spMerger.getMergedClusterND_Host()[y*Kinect::Width+x].x << ", " 
		//									<<spMerger.getMergedClusterND_Host()[y*Kinect::Width+x].y <<", "
		//									<<spMerger.getMergedClusterND_Host()[y*Kinect::Width+x].z << ", "
		//									<<spMerger.getMergedClusterND_Host()[y*Kinect::Width+x].w<<std::endl;
		//	}
		//}
		//save image
		cv::imwrite("input_image.bmp", kinect.GetColorImage());	
		cv::imwrite("random_image.bmp", NASP.getRandomColorImage());
		cv::imwrite("normal_image.bmp", nmg.getNormalImg());
		cv::imwrite("label_image.bmp", spMerger.getSegmentResult());
		//projection
		Projector.PlaneProjection(spMerger.getMergedClusterND_Device(), spMerger.getMergedClusterLabel_Device(), spMerger.getMergedClusterVariance_Device(), points_Device);
		//Projector.PlaneProjection(NASP.getNormalsDevice(), NASP.getCentersDevice(), NASP.getLabelDevice(), NASP.getNormalsVarianceDevice(), points_Device);
		float3* planarPoints_Host;
		cudaMallocHost(&planarPoints_Host, sizeof(float3)*Kinect::Width*Kinect::Height);
		cudaMemcpy(points_Host, points_Device, sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
		cudaMemcpy(planarPoints_Host, Projector.GetPlaneFitted3D_Device(), sizeof(float3)*Kinect::Width*Kinect::Height, cudaMemcpyDeviceToHost);
		for(int y=0; y<Kinect::Height; y++){
			for(int x=0; x<Kinect::Width; x++){
				pcl::PointXYZRGB point;
				if(points_Host[y*Kinect::Width+x].z> 0.0f){
					point.x = points_Host[y*Kinect::Width+x].x/1000.0f;
					point.y = points_Host[y*Kinect::Width+x].y/1000.0f;
					point.z = -points_Host[y*Kinect::Width+x].z/1000.0f;
					point.r = (unsigned char)kinect.GetColorImage().at<cv::Vec3b>(y, x).val[2];
					point.g = (unsigned char)kinect.GetColorImage().at<cv::Vec3b>(y, x).val[1];
					point.b = (unsigned char)kinect.GetColorImage().at<cv::Vec3b>(y, x).val[0];
					input->push_back(point);
				}
				if(planarPoints_Host[y*Kinect::Width+x].z> 50.0f && planarPoints_Host[y*Kinect::Width+x].z< 5000.0f){
					point.x = planarPoints_Host[y*Kinect::Width+x].x/1000.0f;
					point.y = planarPoints_Host[y*Kinect::Width+x].y/1000.0f;
					point.z = -planarPoints_Host[y*Kinect::Width+x].z/1000.0f;
					point.r = (unsigned char)kinect.GetColorImage().at<cv::Vec3b>(y, x).val[2];
					point.g = (unsigned char)kinect.GetColorImage().at<cv::Vec3b>(y, x).val[1];
					point.b = (unsigned char)kinect.GetColorImage().at<cv::Vec3b>(y, x).val[0];
					upsampled->push_back(point);
				}
			}
		}
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_input (new pcl::visualization::PCLVisualizer ("Input Viewer"));
		viewer_input->initCameraParameters ();
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_input(input);
		viewer_input->addPointCloud<pcl::PointXYZRGB> (input, rgb_input, "input");
		viewer_input->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "input");
		
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_upsampled(new pcl::visualization::PCLVisualizer ("Upsample Viewer"));
		viewer_upsampled->initCameraParameters ();
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_upsampled(upsampled);
		viewer_upsampled->addPointCloud<pcl::PointXYZRGB> (upsampled, rgb_upsampled, "upsampled");
		viewer_upsampled->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "upsampled");
		
		bool ShouldRun = true;
		
		while (ShouldRun) {  
			viewer_input->spinOnce (100);
			viewer_upsampled->spinOnce (100);
			boost::this_thread::sleep (boost::posix_time::microseconds (100000));
		}			
	
		}
	}
	return 0;
}

