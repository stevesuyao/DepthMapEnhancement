#include "Projection_GPU.h"
#include "DimensionConvertor.h"
#include "Visualizer.h"

Projection_GPU::Projection_GPU(int width, int height, const cv::Mat intrinsic){
	this->width = width;
	this->height = height;
	//dim = new ApplyRigidTransform();
	dim = new DimensionConvertor();
	dim->setCameraParameters(intrinsic, width, height);
	initMemory();
	initNormalized3D();
}

Projection_GPU::~Projection_GPU(){
	cudaFree(PlaneFitted3D_Host);
	cudaFree(PlaneFitted3D_Device);
	cudaFree(Normalized3D_Device);
	delete dim;

	dim = 0;
}


void Projection_GPU::initMemory(){	
	cudaMallocHost(&PlaneFitted3D_Host, width * height * sizeof(float3));
	cudaMalloc(&PlaneFitted3D_Device, width * height * sizeof(float3));
	cudaMalloc(&Normalized3D_Device, width * height * sizeof(float3));

}
float3*	Projection_GPU::GetPlaneFitted3D_Host()const{
	return PlaneFitted3D_Host;
}

float3*	Projection_GPU::GetPlaneFitted3D_Device()const{
	return PlaneFitted3D_Device;
}
