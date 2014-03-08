#include "Projection_GPU.h"
#include "DimensionConvertor.h"

__global__ void initTemp(float3* temp, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	temp[x + y * width].x = x;
	temp[x + y * width].y = y;
	temp[x + y * width].z = 1;
}

__global__ void setPsuedoDepth(
	const float3* input_3d, 
	float3* plane_fitted, 
	float3* normalized, 
	const float4* nd, 
	const int* labels, 
	const float* variance, 
	int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;


	int l = labels[x + y * width];
	//float a = nd[l].x;
	//float b = nd[l].y;
	//float c = nd[l].z;
	//float d = nd[l].w;
	if(l > -1 && acos(variance[l]) < (3.141592653f / 8.0f)){
		float a = nd[y*width+x].x;
		float b = nd[y*width+x].y;
		float c = nd[y*width+x].z;
		float d = nd[y*width+x].w;

		float3* ref = &plane_fitted[x + y * width];
		ref->z = abs(d / (a * normalized[x + y * width].x + b * normalized[x + y * width].y + c));
		ref->x = ref->z*normalized[x + y * width].x;
		ref->y = ref->z*normalized[x + y * width].y;
	}
	else{
		plane_fitted[x + y * width].x = input_3d[y*width+x].x;
		plane_fitted[x + y * width].y = input_3d[y*width+x].y;
		plane_fitted[x + y * width].z = input_3d[y*width+x].z;
	}
}
__global__ void setPsuedoDepth(
	const float3* input_3d, 
	float3* plane_fitted, 
	float3* normalized, 
	const float3* normals, 
	const float3* centers, 
	const int* labels, 
	const float* variance, 
	int width, int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;


		int l = labels[x + y * width];
		float a = normals[l].x;
		float b = normals[l].y;
		float c = normals[l].z;
		float d = fabs(a*centers[l].x+b*centers[l].y+c*centers[l].z);
		if(l > -1){
			//float a = nd[y*width+x].x;
			//float b = nd[y*width+x].y;
			//float c = nd[y*width+x].z;
			//float d = nd[y*width+x].w;
			if(acos(variance[l]) <  (3.141592653f / 8.0f)){
				float3* ref = &plane_fitted[x + y * width];
				ref->z = abs(d / (a * normalized[x + y * width].x + b * normalized[x + y * width].y + c));
				ref->x = ref->z*normalized[x + y * width].x;
				ref->y = ref->z*normalized[x + y * width].y;
			}
			else{
				plane_fitted[x + y * width].x = input_3d[y*width+x].x;
				plane_fitted[x + y * width].y = input_3d[y*width+x].y;
				plane_fitted[x + y * width].z = input_3d[y*width+x].z;
			}
		}
		else{
			plane_fitted[x + y * width].x = input_3d[y*width+x].x;
			plane_fitted[x + y * width].y = input_3d[y*width+x].y;
			plane_fitted[x + y * width].z = input_3d[y*width+x].z;
		}
}
//void Projection_GPU::getProjectedMap(){
//	//initialize
//	initTemp<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//		(Normalized3D_Device, width, height);
//	//prepare for projection
//	dim->projectiveToReal(Normalized3D_Device, Normalized3D_Device, width*height);
//	//plane projection
//	setPsuedoDepth<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//		(Upsampled3D_Device, Normalized3D_Device, normal_device, labels_device, width, height);
//	
//}

void Projection_GPU::initNormalized3D(){
	//initialize
	initTemp<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(Normalized3D_Device, width, height);
	//prepare for projection
	dim->projectiveToReal(Normalized3D_Device, Normalized3D_Device);
}
__global__ void getFinalizedOutputKernel(const float3* input3d, float3* planefitted3d,
	//const float4* nd,
	float3* normalized_3d,
	const int* labels,
	int width,
	int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int access = x + y * width;
		int l = labels[access];

		if(l == -1){
			//“_‚Ì”‚Å”»’f
			planefitted3d[access].x = input3d[access].x;
			planefitted3d[access].y = input3d[access].y;
			planefitted3d[access].z = input3d[access].z;
		}

		//Interpolation‚Æ‚Ì”äŠr
		else if(input3d[access].z > 50.0){
			float distance = sqrt(pow(planefitted3d[access].x-input3d[access].x, 2) +
				pow(planefitted3d[access].y-input3d[access].y, 2) +
				pow(planefitted3d[access].z-input3d[access].z, 2));
			//•½–Ê‚É‚·‚é‚Æ‚¸‚ê‚ª‘å‚«‚¢“_i•½–Ê‚É‚Í‚Å‚«‚È‚¢j
			if(distance >= 500.0){
				///•½‹Ï‚Æ‚Á‚Ä‚à‚¢‚¢
				/*planefitted3d[access].x = (planefitted3d[access].x+input3d[access].x)/2.0;
				planefitted3d[access].y = (planefitted3d[access].y+input3d[access].y)/2.0;
				planefitted3d[access].z = (planefitted3d[access].z+input3d[access].z)/2.0;*/
				planefitted3d[access].x = input3d[access].x;
				planefitted3d[access].y = input3d[access].y;
				planefitted3d[access].z = input3d[access].z;
			}
			else if(distance > 300.0){
				planefitted3d[access].x = (planefitted3d[access].x+input3d[access].x)/2.0;
				planefitted3d[access].y = (planefitted3d[access].y+input3d[access].y)/2.0;
				planefitted3d[access].z = (planefitted3d[access].z+input3d[access].z)/2.0;
				/*planefitted3d[access].z = ((7.0-distance)*planefitted3d[access].z+(distance-3.0)*input3d[access].z)/4.0;
				planefitted3d[access].x = ((7.0-distance)*planefitted3d[access].x+(distance-3.0)*input3d[access].x)/4.0;
				planefitted3d[access].y = ((7.0-distance)*planefitted3d[access].y+(distance-3.0)*input3d[access].y)/4.0;*/
			}
		}
		//Interpolation‚Ì“_‚ª‚È‚¢‚Æ‚«
		else {
			//Occlusion‚©ƒmƒCƒY‚©‚ð”»’f
			int count_neighber = 0; 
			//scan right
			float3 right_point;
			right_point.x = 0.0;
			right_point.y = 0.0;
			right_point.z = 0.0;
			int scan_right = 0;
			bool rightExist = false;
			while(x+scan_right < width && 
				labels[access+scan_right]==l &&
				rightExist==false){
					scan_right++;			
					if(input3d[access+scan_right].z > 50){
						right_point.x = input3d[access+scan_right].x;
						right_point.y = input3d[access+scan_right].y;
						right_point.z = input3d[access+scan_right].z;
						rightExist = true;
						count_neighber++;
					}
			};

			//scan left
			float3 left_point;
			int scan_left = 0;
			left_point.x = 0.0;
			left_point.y = 0.0;
			left_point.z = 0.0;
			bool leftExist = false;
			while(x-scan_left >= 0 && 
				labels[access-scan_left]==l &&
				leftExist == false){
					scan_left++;
					if(input3d[access-scan_left].z > 50){
						left_point.x = input3d[access-scan_left].x;
						left_point.y = input3d[access-scan_left].y;
						left_point.z = input3d[access-scan_left].z;
						leftExist = true;
						count_neighber++;
					}
					scan_left++;
			};
			//scan up
			float3 up_point;
			up_point.x = 0.0;
			up_point.y = 0.0;
			up_point.z = 0.0;
			int scan_up = 0;
			bool upExist = false;
			while(y-scan_up >= 0 && 
				labels[access-scan_up*width]==l &&
				upExist == false){
					scan_up++;
					if(input3d[access-scan_up*width].z > 50){
						up_point.x = input3d[access-scan_up*width].x;
						up_point.y = input3d[access-scan_up*width].y;
						up_point.z = input3d[access-scan_up*width].z;
						upExist = true;
						count_neighber++;
					}
			};
			//scan down
			float3 down_point;
			down_point.x = 0.0;
			down_point.y = 0.0;
			down_point.z = 0.0;
			int scan_down = 0;
			bool downExist = false;
			while(y+scan_down < height && 
				labels[access+scan_down*width]==l &&
				downExist == false){
					scan_down++;
					if(input3d[access+scan_down*width].z > 50){
						down_point.x = input3d[access+scan_down*width].x;
						down_point.y = input3d[access+scan_down*width].y;
						down_point.z = input3d[access+scan_down*width].z;
						downExist = true;
						count_neighber++;
					}
			};
			//Occulusion‚Ì‚Æ‚«
			if(count_neighber < 2){
				planefitted3d[access].x = 0.0;
				planefitted3d[access].y = 0.0;
				planefitted3d[access].z = 0.0;
			}
			//Occulusion‚Å‚Í‚È‚¢‚Æ‚«
			else{
				//interpolation‚ðƒNƒ‰ƒXƒ^“à‚Å‚¨‚±‚È‚¤
				//‰¡•ûŒü‚Ìinterpolation
				float3 interpolate_horizontal;
				if(rightExist && leftExist){
					interpolate_horizontal.x = (left_point.x*scan_right + right_point.x*scan_left)/(float)(scan_right+scan_left); 
					interpolate_horizontal.y = (left_point.y*scan_right + right_point.y*scan_left)/(float)(scan_right+scan_left);
					interpolate_horizontal.z = (left_point.z*scan_right + right_point.z*scan_left)/(float)(scan_right+scan_left);
				}
				else{
					interpolate_horizontal.x = left_point.x + right_point.x;
					interpolate_horizontal.y = left_point.y + right_point.y;
					interpolate_horizontal.z = left_point.z + right_point.z;
				}
				//c•ûŒü‚Ìinterpolation
				float3 interpolate_vertical;
				if(upExist && downExist){
					interpolate_vertical.x = (up_point.x*scan_down + down_point.x*scan_up)/(float)(scan_up+scan_down); 
					interpolate_vertical.y = (up_point.y*scan_down + down_point.y*scan_up)/(float)(scan_up+scan_down);
					interpolate_vertical.z = (up_point.z*scan_down + down_point.z*scan_up)/(float)(scan_up+scan_down);
				}
				else{
					interpolate_vertical.x = up_point.x + down_point.x;
					interpolate_vertical.y = up_point.y + down_point.y;
					interpolate_vertical.z = up_point.z + down_point.z;
				}
				//interpolation
				float3 interpolate;
				if(interpolate_horizontal.z > 50.0 && interpolate_vertical.z > 50.0){
					//interpolate.x = (interpolate_horizontal.x + interpolate_vertical.x) / 2.0;
					//interpolate.y = (interpolate_horizontal.y + interpolate_vertical.y) / 2.0;
					interpolate.z = (interpolate_horizontal.z + interpolate_vertical.z) / 2.0;
					interpolate.x = interpolate.z * normalized_3d[access].x;
					interpolate.y = interpolate.z * normalized_3d[access].y;
				}
				else{
					//interpolate.x = interpolate_horizontal.x + interpolate_vertical.x;
					//interpolate.y = interpolate_horizontal.y + interpolate_vertical.y;
					interpolate.z = interpolate_horizontal.z + interpolate_vertical.z;
					interpolate.x = interpolate.z * normalized_3d[access].x;
					interpolate.y = interpolate.z * normalized_3d[access].y; 
				}
				//•½–Ê‚Æinterpolation‚Ì·‚ðŒ©‚é
				float distance = sqrt(pow(planefitted3d[access].x-interpolate.x, 2) +
					pow(planefitted3d[access].y-interpolate.y, 2) +
					pow(planefitted3d[access].z-interpolate.z, 2));
				if(distance > 500.0){
					if(count_neighber == 2){
						planefitted3d[access].x = 0.0;
						planefitted3d[access].y = 0.0;
						planefitted3d[access].z = 0.0;
					}
					else{
						planefitted3d[access].x = interpolate.x;
						planefitted3d[access].y = interpolate.y;
						planefitted3d[access].z = interpolate.z;
					}
					/*planefitted3d[access].x = 0.0;
					planefitted3d[access].y = 0.0;
					planefitted3d[access].z = 0.0;*/
				}
				else if(distance > 300.0){
					planefitted3d[access].x = interpolate.x;
					planefitted3d[access].y = interpolate.y;
					planefitted3d[access].z = interpolate.z;
				}
				else if(distance > 200.0){
					planefitted3d[access].x = (planefitted3d[access].x + interpolate.x)/2.0;
					planefitted3d[access].y = (planefitted3d[access].y + interpolate.y)/2.0;
					planefitted3d[access].z = (planefitted3d[access].z + interpolate.z)/2.0;
				}
			}
		}
}



__device__ void _atomicMin(double* address, double* val){
	double old = *address, assumed;
	do{
		assumed = old;
		old = 
			__longlong_as_double(
			atomicCAS(
			(unsigned long long int*)address, 
			__double_as_longlong(assumed), 
			__double_as_longlong(
			(*((float*)val) > *((float*)&assumed)) ? assumed : *val				
			)
			)
			);

	}while(assumed != old);
}


__global__ void makeJustifiedDepthMap(const float3* interpolate3D, const float2* interpolate_img, float3* depth_out, int width, int height){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	depth_out[x + y * width].x=0.0;
	depth_out[x + y * width].y=0.0;
	depth_out[x + y * width].z=0.0;

	int2 j_coordinate;
	j_coordinate.x = (int)(interpolate_img[x + y * width].x+0.5);
	j_coordinate.y = (int)(interpolate_img[x + y * width].y+0.5);

	if(j_coordinate.x >= 0.0 && j_coordinate.y >= 0.0 &&
		j_coordinate.x < width && j_coordinate.y < height){

			//_atomicMin((double*)&depth_out[j_coordinate.x + j_coordinate.y * width], (double*)&interpolate3D[x + y * width].z);
			depth_out[j_coordinate.x + j_coordinate.y * width].x = interpolate3D[x + y * width].x;
			depth_out[j_coordinate.x + j_coordinate.y * width].y = interpolate3D[x + y * width].y;
			depth_out[j_coordinate.x + j_coordinate.y * width].z = interpolate3D[x + y * width].z;
	}
	//depth_out[j_coordinate.x + j_coordinate.y * width] = interpolate3D[x + y * width].z
}
__global__ void mrf_optimization(
	const float3* input3d,
	float3* planefitted3d,
	float3* normalized_3d,
	const int* labels,
	int width,
	int height,
	int window_size,
	float K,
	float smooth_sigma){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(planefitted3d[y*width+x].z > 50.0f){
		//mrf optimization
		float numerator = planefitted3d[y*width+x].z, denominator = 1.0f;
		for(int i = - window_size/2; i <= window_size/2; i++){		// y
			for(int j = -window_size/2; j <= window_size/2; j++){		// x
				int xj = x+j, yi = y+i;
				if(xj >= 0 && xj < width && yi >= 0 && yi < height && input3d[yi*width+xj].z > 50.0f ){
					//float distance = sqrt(pow(planefitted3d[y*width+x].x-input3d[y*width+x].x, 2) +
					//						pow(planefitted3d[y*width+x].y-input3d[y*width+x].y, 2) +
					//							pow(planefitted3d[y*width+x].z-input3d[y*width+x].z, 2));
					float diff = fabs(input3d[y*width+x].z-input3d[yi*width+xj].z);
					float depth_filter = K/(1+pow(diff, 2.0f));
					//calculate filter
					float filter = smooth_sigma*depth_filter;
					numerator += input3d[yi*width+xj].z*filter; 
					denominator += filter;
				}
			}
		}
		if(denominator != 0.0f){
			float depth = numerator/denominator;	
			planefitted3d[y*width+x].z = depth;
			planefitted3d[y*width+x].x = normalized_3d[y*width+x].x*depth;
			planefitted3d[y*width+x].y = normalized_3d[y*width+x].y*depth;
			}
		}
}
__global__ void variance_optimization(
	const float3* input3d,
	const float* variance, 
	float3* planefitted3d,
	float3* normalized_3d,
	const int* labels,
	int width,
	int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if(planefitted3d[y*width+x].z > 50.0f){
				if(fabs((float)planefitted3d[y*width+x].z-(float)input3d[y*width+x].z)>input3d[y*width+x].z*0.01f){
						planefitted3d[y*width+x].x = input3d[y*width+x].x;
						planefitted3d[y*width+x].y = input3d[y*width+x].y;
						planefitted3d[y*width+x].z = input3d[y*width+x].z;
				}
				else if(labels[y*width+x] > -1 && (acos(variance[labels[y*width+x]]) < (3.141592653f / 8.0f))){
						planefitted3d[y*width+x].z = planefitted3d[y*width+x].z*variance[labels[y*width+x]]+input3d[y*width+x].z*(1.0f-variance[labels[y*width+x]]);
						//planefitted3d[y*width+x].z = planefitted3d[y*width+x].z*(1.0f-variance[y*width+x])+input3d[y*width+x].z*variance[y*width+x];
						planefitted3d[y*width+x].x = normalized_3d[y*width+x].x*planefitted3d[y*width+x].z;
						planefitted3d[y*width+x].y = normalized_3d[y*width+x].y*planefitted3d[y*width+x].z;
				}
		
		}
}
void Projection_GPU::PlaneProjection(const float4* nd_device, const int* labels_device, const float* variance_device, const float3* points3d_device){
	//‚·‚×‚Ä‚Ì“_‚ð•½–Êfitting‚·‚é
	//getProjectedMap();
	//plane projection
	setPsuedoDepth<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, nd_device, labels_device, variance_device, width, height);

	//Input‚Æ‚Ì”äŠr
	//getFinalizedOutputKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//	(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height);
	//mrf_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 5, 0.5f, 1.0f);
	variance_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(points3d_device, variance_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height);

	//3D¨2D
	//dim->realToProjective2(PlaneFitted3D_Device, Upsampled2D_Device, width*height);
	//Device to Host
	cudaMemcpy(PlaneFitted3D_Host, PlaneFitted3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
	//for(int y=0; y<height; y++){
	//	for(int x=0; x<width; x++){
	//		std::cout << PlaneFitted3D_Host[y*width+x].z <<std::endl;
	//	}
	//}
}

void Projection_GPU::PlaneProjection(
	const float3* normals_device, 
	const float3* centers_device, 
	const int* labels_device,
	const float* variance_device, 
	const float3* points3d_device){	
		//‚·‚×‚Ä‚Ì“_‚ð•½–Êfitting‚·‚é
		//getProjectedMap();
		//plane projection
		setPsuedoDepth<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, 
			normals_device, centers_device, labels_device, variance_device, width, height);

		//Input‚Æ‚Ì”äŠr
		//getFinalizedOutputKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		//	(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height);
		mrf_optimization<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(points3d_device, PlaneFitted3D_Device, Normalized3D_Device, labels_device, width, height, 15, 200.0f, 200.0f);

		//3D¨2D
		//dim->realToProjective2(PlaneFitted3D_Device, Upsampled2D_Device, width*height);
		//Device to Host
		cudaMemcpy(PlaneFitted3D_Host, PlaneFitted3D_Device, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
		//for(int y=0; y<height; y++){
		//	for(int x=0; x<width; x++){
		//		std::cout << PlaneFitted3D_Host[y*width+x].z <<std::endl;
		//	}
		//}
}