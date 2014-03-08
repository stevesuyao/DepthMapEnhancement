#include "LabelEquivalenceSeg.h"
#include <cuda_runtime.h>
#include <thrust\fill.h>
#include <thrust\device_ptr.h>
#include <iostream>

//__global__ void initLabel(float4* input_nd, int* merged_cluster_label, int* ref, 
//	float4*  cluster_nd, int* cluster_label, int width, int height){
//		int x = blockIdx.x * blockDim.x + threadIdx.x;
//		int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//		
//		if(abs(cluster_nd[cluster_label[x+y*width]].x) < 1.1 && cluster_label[x+y*width] > -1){
//			input_nd[x+y*width] = cluster_nd[cluster_label[x+y*width]];
//			//merged_cluster_label[x+y*width] = x+y*width;	
//			ref[x+y*width] = x+y*width;
//			//ref[x+y*width] = cluster_label[x+y*width];
//			//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
//			merged_cluster_label[x+y*width] = cluster_label[x+y*width];	
//			
//		}
//		else {
//			input_nd[x+y*width].x = 5.0;
//			input_nd[x+y*width].y = 5.0;
//			input_nd[x+y*width].z = 5.0;
//			input_nd[x+y*width].w = 5.0;
//			input_nd[x+y*width] = cluster_nd[cluster_label[x+y*width]];
//			ref[x+y*width] = x+y*width;
//			//merged_cluster_label[x+y*width] = x+y*width;
//			//ref[x+y*width] = cluster_label[x+y*width];
//			//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
//			//merged_cluster_label[x+y*width] = cluster_label[x+y*width];	
//			merged_cluster_label[x+y*width] = -1;
//			//ref[x+y*width] = -1;
//		}
//}

__global__ void initLabel(float4* input_nd, int* merged_cluster_label, int* ref, 
	float3*  cluster_normals, int* cluster_label, float3* cluster_centers, int width, int height){
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		
		if(cluster_normals[cluster_label[x+y*width]].x != -1.0f || 
				cluster_normals[cluster_label[x+y*width]].y !=  -1.0f ||
					cluster_normals[cluster_label[x+y*width]].z != -1.0f){
			input_nd[x+y*width].x = cluster_normals[cluster_label[x+y*width]].x;
			input_nd[x+y*width].y = cluster_normals[cluster_label[x+y*width]].y;
			input_nd[x+y*width].z = cluster_normals[cluster_label[x+y*width]].z;
			input_nd[x+y*width].w = fabs(cluster_normals[cluster_label[x+y*width]].x*cluster_centers[cluster_label[x+y*width]].x + 
											cluster_normals[cluster_label[x+y*width]].y*cluster_centers[cluster_label[x+y*width]].y +
												cluster_normals[cluster_label[x+y*width]].z*cluster_centers[cluster_label[x+y*width]].z);
			ref[x+y*width] = x+y*width;
			merged_cluster_label[x+y*width] = cluster_label[x+y*width];
			//merged_cluster_label[x+y*width] = x+y*width;

			//ref[x+y*width] = cluster_label[x+y*width];
			//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
			//merged_cluster_label[x+y*width] = cluster_label[x+y*width];	
			//ref[x+y*width] = cluster_label[x+y*width];
			
		}
		else {
			input_nd[x+y*width].x = 5.0;
			input_nd[x+y*width].y = 5.0;
			input_nd[x+y*width].z = 5.0;
			input_nd[x+y*width].w = 5.0;
			ref[x+y*width] = x+y*width;
			merged_cluster_label[x+y*width] = -1;

			//merged_cluster_label[x+y*width] = x+y*width;
			//ref[x+y*width] = cluster_label[x+y*width];
			//ref[cluster_label[x+y*width]] = x+y*width;
			//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
			
			//merged_cluster_label[x+y*width] = cluster_label[x+y*width];	
		
			
			//ref[x+y*width] = -1;
		}
}
//__global__ void initLabel(float4* input_nd, int* merged_cluster_label, int* ref, LabelEquivalenceSeg::rgb* sp_color,
//							float3*  cluster_normals, int* cluster_label, float3* cluster_centers, SuperpixelSegmentation::superpixel* sp_data, 
//			int width, int height){
//		int x = blockIdx.x * blockDim.x + threadIdx.x;
//		int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//		sp_color[x+y*width].r = sp_data[cluster_label[x+y*width]].r;
//		sp_color[x+y*width].g = sp_data[cluster_label[x+y*width]].g;
//		sp_color[x+y*width].b = sp_data[cluster_label[x+y*width]].b;
//
//		if(cluster_normals[cluster_label[x+y*width]].x != -1.0f || 
//				cluster_normals[cluster_label[x+y*width]].y !=  -1.0f ||
//					cluster_normals[cluster_label[x+y*width]].z != -1.0f){
//			input_nd[x+y*width].x = cluster_normals[cluster_label[x+y*width]].x;
//			input_nd[x+y*width].y = cluster_normals[cluster_label[x+y*width]].y;
//			input_nd[x+y*width].z = cluster_normals[cluster_label[x+y*width]].z;
//			input_nd[x+y*width].w = cluster_centers[cluster_label[x+y*width]].z;
//			ref[x+y*width] = x+y*width;
//			//merged_cluster_label[x+y*width] = x+y*width;	
//			//atomicMin(&ref[x+y*width], cluster_label[x+y*width]);
//			//atomicMin(&merged_cluster_label[x+y*width], cluster_label[x+y*width]);
//			//ref[x+y*width] = cluster_label[x+y*width];
//			merged_cluster_label[x+y*width] = cluster_label[x+y*width];
//		} 
//		else {
//			input_nd[x+y*width].x = 5.0;
//			input_nd[x+y*width].y = 5.0;
//			input_nd[x+y*width].z = 5.0;
//			input_nd[x+y*width].w = 5.0;
//			ref[x+y*width] = x+y*width;
//			//merged_cluster_label[x+y*width] = x+y*width;	
//			//atomicMin(&ref[x+y*width], cluster_label[x+y*width]);
//			//atomicMin(&merged_cluster_label[x+y*width], cluster_label[x+y*width]);
//			//ref[x+y*width] = cluster_label[x+y*width];
//			//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
//			//merged_cluster_label[x+y*width] = cluster_label[x+y*width];	
//			merged_cluster_label[x+y*width] = -1;
//			//ref[x+y*width] = -1;
//		}
//}

//template<int blockSize>
//__global__ void initLabel(float4* input_nd, int* merged_cluster_label, int* ref, LabelEquivalenceSeg::rgb* sp_color,
//							float3*  cluster_normals, int* cluster_label, float3* cluster_centers, SuperpixelSegmentation::superpixel* sp_data, 
//								int rows, int cols, int width, int height){
//		__shared__ int label_shared[blockSize];
//		//thread id
//		int tid = threadIdx.y*blockDim.x+threadIdx.x;
//		//current cluster
//		int2 cluster_pos;
//		cluster_pos.x = blockIdx.x;
//		cluster_pos.y = blockIdx.y;
//		int cluster_id = cluster_pos.y*cols+cluster_pos.x;
//		//assign threads around cluster
//		int2 arounds;
//		int2 ref_pixels;
//		ref_pixels.x = (width/cols)*8/blockDim.x+1;
//		ref_pixels.y = (height/rows)*8/blockDim.y+1;
//		label_shared[tid] = 999999;
//		for(int yy=0; yy<ref_pixels.y; yy++){
//			for(int xx=0; xx<ref_pixels.x; xx++){
//				arounds.x = sp_data[cluster_id].x+(threadIdx.x-blockDim.x/2)*ref_pixels.x+xx;
//				arounds.y = sp_data[cluster_id].y+(threadIdx.y-blockDim.y/2)*ref_pixels.y+yy;
//				if(arounds.x>=0 && arounds.x<width && arounds.y>=0 && arounds.y<height){
//					int around_id = cluster_label[arounds.y*width+arounds.x];
//					if(around_id == cluster_id){
//						if(label_shared[tid]>arounds.y*width+arounds.x)
//							label_shared[tid] = arounds.y*width+arounds.x;
//					}
//				}
//			}
//		}
//		__syncthreads();
//		//calculate min label
//		if(blockSize >= 1024){
//			if(tid < 512){
//				if(label_shared[tid] > label_shared[tid+512]) 
//					label_shared[tid] = label_shared[tid+512];
//			}
//			__syncthreads();
//		}
//		if(blockSize >= 512){
//			if(tid < 256){
//				if(label_shared[tid] > label_shared[tid+256]) 
//					label_shared[tid] = label_shared[tid+256];
//			}
//			__syncthreads();
//		}
//		if(blockSize >= 256){
//			if(tid < 128){
//				if(label_shared[tid] > label_shared[tid+128]) 
//					label_shared[tid] = label_shared[tid+128];
//			}
//			__syncthreads();
//		}
//		if(blockSize >= 128){
//			if(tid < 64){
//				if(label_shared[tid] > label_shared[tid+64]) 
//					label_shared[tid] = label_shared[tid+64];
//			}
//			__syncthreads();
//		}
//		if(tid < 32){
//			if(blockSize >= 64){
//				if(label_shared[tid] > label_shared[tid+32]) 
//					label_shared[tid] = label_shared[tid+32];
//			}
//			if(blockSize >= 32){
//				if(label_shared[tid] > label_shared[tid+16]) 
//					label_shared[tid] = label_shared[tid+16];
//			}
//			if(blockSize >= 16){
//				if(label_shared[tid] > label_shared[tid+8]) 
//					label_shared[tid] = label_shared[tid+8];
//			}
//			if(blockSize >= 8){
//				if(label_shared[tid] > label_shared[tid+4]) 
//					label_shared[tid] = label_shared[tid+4];
//			}
//			if(blockSize >= 4){
//				if(label_shared[tid] > label_shared[tid+2]) 
//					label_shared[tid] = label_shared[tid+2];
//			}
//			if(blockSize >= 2){
//				if(label_shared[tid] > label_shared[tid+1]) 
//					label_shared[tid] = label_shared[tid+1];
//			}
//		}
//		__syncthreads();
//		//store data
//		for(int yy=0; yy<ref_pixels.y; yy++){
//			for(int xx=0; xx<ref_pixels.x; xx++){
//				arounds.x = sp_data[cluster_id].x+(threadIdx.x-blockDim.x/2)*ref_pixels.x+xx;
//				arounds.y = sp_data[cluster_id].y+(threadIdx.y-blockDim.y/2)*ref_pixels.y+yy;
//				if(arounds.x>=0 && arounds.x<width && arounds.y>=0 && arounds.y<height){
//					int around_id = cluster_label[arounds.y*width+arounds.x];
//					if(around_id == cluster_id){
//							sp_color[arounds.y*width+arounds.x].r = sp_data[around_id].r;
//							sp_color[arounds.y*width+arounds.x].g = sp_data[around_id].g;
//							sp_color[arounds.y*width+arounds.x].b = sp_data[around_id].b;
//
//							if(cluster_normals[around_id].x != -1.0f || 
//									cluster_normals[around_id].y !=  -1.0f ||
//										cluster_normals[around_id].z != -1.0f){
//								input_nd[arounds.y*width+arounds.x].x = cluster_normals[around_id].x;
//								input_nd[arounds.y*width+arounds.x].y = cluster_normals[around_id].y;
//								input_nd[arounds.y*width+arounds.x].z = cluster_normals[around_id].z;
//								input_nd[arounds.y*width+arounds.x].w = cluster_centers[around_id].z;
//								ref[arounds.y*width+arounds.x] = label_shared[0];
//								//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
//								merged_cluster_label[arounds.y*width+arounds.x] =label_shared[0];	
//								//ref[x+y*width] = cluster_label[x+y*width];
//								//merged_cluster_label[x+y*width] = cluster_label[x+y*width];
//							} 
//							else {
//								input_nd[arounds.y*width+arounds.x].x = 5.0;
//								input_nd[arounds.y*width+arounds.x].y = 5.0;
//								input_nd[arounds.y*width+arounds.x].z = 5.0;
//								input_nd[arounds.y*width+arounds.x].w = 5.0;
//								ref[arounds.y*width+arounds.x] =label_shared[0];
//								merged_cluster_label[arounds.y*width+arounds.x] = label_shared[0];	
//								//ref[x+y*width] = cluster_label[x+y*width];
//								//atomicMin(&ref[cluster_label[x+y*width]], x+y*width);
//								//merged_cluster_label[x+y*width] = cluster_label[x+y*width];	
//								//merged_cluster_label[x+y*width] = -1;
//								//ref[x+y*width] = -1;
//							}
//					
//					}
//				}
//			}
//		}
//
//}
__device__ bool compNormal(float4* a, float4* b){
	return 
		acos(a->x * b->x + a->y * b->y + a->z * b->z)>0 && 
		acos(a->x * b->x + a->y * b->y + a->z * b->z) < (3.141592653f / 8.0f)
		&&
		abs(a->w - b->w) < 30.0f;
}


__device__ int getMin(
	float4& up_nd,
	int& merged_up_label,
	int up_label,
	float4& left_nd,
	int& merged_left_label,
	int left_label,
	float4& center_nd,
	int& merged_center_label,
	int center_label,
	float4& right_nd,
	int& merged_right_label,
	int right_label,
	float4& down_nd,
	int& merged_down_label,
	int down_label){
		int c;
		c = merged_up_label > -1 && (up_label==center_label || compNormal(&up_nd, &center_nd)) && merged_up_label < merged_center_label ? merged_up_label : merged_center_label;
		c = merged_left_label > -1 && (left_label==center_label || compNormal(&left_nd, &center_nd)) && merged_left_label < c ? merged_left_label : c;
		c = merged_right_label > -1 && (right_label==center_label || compNormal(&right_nd, &center_nd)) &&merged_right_label < c ? merged_right_label : c;
		return merged_down_label > -1 && (down_label==center_label || compNormal(&down_nd, &center_nd)) && merged_down_label < c ? merged_down_label : c;
}


__global__ void scanKernel(
	float4* input_nd, 
	int* merged_cluster_label, 
	int* ref, 
	int* cluster_label, 
	float* variance,
	int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	int label1 = merged_cluster_label[x + y * width];
	int label2;

	if(label1 > -1 && acos(variance[label1]) < (3.141592653f / 3.0f)){
		label2 = getMin(
			//up
			input_nd[x + (y - 1 > 0 ? y - 1 : 0) * width],
			merged_cluster_label[x + (y - 1 > 0 ? y - 1 : 0) * width],
			cluster_label[x + (y - 1 > 0 ? y - 1 : 0) * width], 
			//left
			input_nd[(x - 1 > 0 ? x - 1 : 0) + y * width],
			merged_cluster_label[(x - 1 > 0 ? x - 1 : 0) + y * width],
			cluster_label[(x - 1 > 0 ? x - 1 : 0) + y * width],
			//center
			input_nd[x + y * width],
			merged_cluster_label[x + y * width],
			cluster_label[x + y*width],
			//right
			input_nd[(x + 1 < width ? x + 1 : width) + y * width],
			merged_cluster_label[(x + 1 < width ? x + 1 : width) + y * width],
			cluster_label[(x + 1 < width ? x + 1 : width) + y * width],
			//down
			input_nd[x + (y + 1 < height ? y + 1 : height) * width],
			merged_cluster_label[x + (y + 1 < height ? y + 1 : height) * width],
			cluster_label[x + (y + 1 < height ? y + 1 : height) * width]);
	
		if(label2 < label1){
			atomicMin(&ref[label1], label2);
		}	
	}
}
//__device__ bool compNormalRGB(float4* a, float4* b, LabelEquivalenceSeg::rgb* a_rgb, LabelEquivalenceSeg::rgb* b_rgb){
//	return 
//		//(abs(a->x) < 1.1 && abs(b->x) < 1.1 && 
//		//abs(acos(a->x * b->x + a->y * b->y 
//		//+ a->z * b->z)) < (3.141592653f / 8.0f)
//		//&&
//		//abs(a->w - b->w) < 700.0f) 
//		//||
//		//((abs((int)a_rgb->r-(int)b_rgb->r) +
//		//	abs((int)a_rgb->g-(int)b_rgb->g) +
//		//		abs((int)a_rgb->b-(int)b_rgb->b)) < 5);
//		abs(a->x) < 1.1 && abs(b->x) < 1.1 && (
//		(abs(acos(a->x * b->x + a->y * b->y 
//		+ a->z * b->z)) < (3.141592653f / 8.0f)
//		&&
//		abs(a->w - b->w) < 500.0f) );
//		//||
//		//((abs((float)a_rgb->r-(float)b_rgb->r) +
//		//	abs((float)a_rgb->g-(float)b_rgb->g) +
//		//		abs((float)a_rgb->b-(float)b_rgb->b))/**
//		//		(1.0f-(a->x * b->x + a->y * b->y + a->z * b->z))*/ < 15.0f));
//
//}
//
//
//__device__ int getMin(
//	float4& up_nd,
//	int& merged_up_label,
//	int up_label,
//	LabelEquivalenceSeg::rgb& up_rgb,
//	float4& left_nd,
//	int& merged_left_label,
//	int left_label,
//	LabelEquivalenceSeg::rgb& left_rgb,
//	float4& center_nd,
//	int& merged_center_label,
//	int center_label,
//	LabelEquivalenceSeg::rgb& center_rgb,
//	float4& right_nd,
//	int& merged_right_label,
//	int right_label,
//	LabelEquivalenceSeg::rgb& right_rgb,
//	float4& down_nd,
//	int& merged_down_label,
//	int down_label,
//	LabelEquivalenceSeg::rgb& down_rgb){
//		int c;
//		c = merged_up_label > -1 && (up_label==center_label || compNormalRGB(&up_nd, &center_nd, &up_rgb, &center_rgb)) && merged_up_label < merged_center_label ? merged_up_label : merged_center_label;
//		c = merged_left_label > -1 && (left_label==center_label || compNormalRGB(&left_nd, &center_nd, &left_rgb, &center_rgb)) && merged_left_label < c ? merged_left_label : c;
//		c = merged_right_label > -1 && (right_label==center_label || compNormalRGB(&right_nd,  &center_nd, &right_rgb, &center_rgb)) &&merged_right_label < c ? merged_right_label : c;
//		return merged_down_label > -1 && (down_label==center_label || compNormalRGB(&down_nd,  &center_nd, &down_rgb, &center_rgb)) && merged_down_label < c ? merged_down_label : c;
//}


//__global__ void scanKernel(float4* input_nd, int* merged_cluster_label, LabelEquivalenceSeg::rgb* sp_data, int* ref, int* cluster_label, int width, int height){
//	int x = (blockIdx.x * blockDim.x + threadIdx.x);
//	int y = (blockIdx.y * blockDim.y + threadIdx.y);
//	int label1 = merged_cluster_label[x + y * width];
//	int label2;
//
//	if(label1 > -1){
//		label2 = getMin(
//			//up
//			input_nd[x + (y - 1 > 0 ? y - 1 : 0) * width],
//			merged_cluster_label[x + (y - 1 > 0 ? y - 1 : 0) * width],
//			cluster_label[x + (y - 1 > 0 ? y - 1 : 0) * width], 
//			sp_data[x + (y - 1 > 0 ? y - 1 : 0) * width], 
//			//left
//			input_nd[(x - 1 > 0 ? x - 1 : 0) + y * width],
//			merged_cluster_label[(x - 1 > 0 ? x - 1 : 0) + y * width],
//			cluster_label[(x - 1 > 0 ? x - 1 : 0) + y * width],
//			sp_data[(x - 1 > 0 ? x - 1 : 0) + y * width], 
//			//center
//			input_nd[x + y * width],
//			merged_cluster_label[x + y * width],
//			cluster_label[x + y*width],
//			sp_data[x + y*width], 
//			//right
//			input_nd[(x + 1 < width ? x + 1 : width) + y * width],
//			merged_cluster_label[(x + 1 < width ? x + 1 : width) + y * width],
//			cluster_label[(x + 1 < width ? x + 1 : width) + y * width],
//			sp_data[(x + 1 < width ? x + 1 : width) + y * width],
//			//down
//			input_nd[x + (y + 1 < height ? y + 1 : height) * width],
//			merged_cluster_label[x + (y + 1 < height ? y + 1 : height) * width],
//			cluster_label[x + (y + 1 < height ? y + 1 : height) * width],
//			sp_data[x + (y + 1 < height ? y + 1 : height) * width]);
//	
//		if(label2 < label1){
//			atomicMin(&ref[label1], label2);
//		}	
//	}
//}

__global__ void analysisKernel(
	int* merged_cluster_label, 
	int* ref, 
	int* cluster_label,
	/*float3* normals,
	float3* centers,*/
	int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	if(merged_cluster_label[x+y*width] ==  cluster_label[x+y*width]){
	//if(merged_cluster_label[x+y*width] ==  x+y*width){
		//label...¡‚ÌêŠ  //current...‚»‚ÌêŠ‚Ìlabel
		int current = ref[x+y*width];
		//‚»‚ÌêŠ‚Ìlabel‚ª‚Â‚¢‚½—Ìˆæ‚ÅêŠ‚Ì’l(x+y*width)‚Ælabel‚ªˆê’v‚·‚éêŠ‚ð’Tõ
		do{
			current = ref[current];
		} while(current != ref[current]);
		//’Tõ‚µ‚½label(‚Â‚Ü‚è‚»‚Ìlabel”Ô†‚ª‚Â‚¢‚½êŠ‚Ì’l)
		ref[x+y*width] = current;		
	}
	//Labeling phase
	if(merged_cluster_label[x+y*width]> -1){
		//float3 normal, center;
		//normal.x = normals[merged_cluster_label[x+y*width]].x;
		//normal.y = normals[merged_cluster_label[x+y*width]].y;
		//normal.z = normals[merged_cluster_label[x+y*width]].z;
		//center.x = centers[merged_cluster_label[x+y*width]].x;
		//center.y = centers[merged_cluster_label[x+y*width]].y;
		//center.z = centers[merged_cluster_label[x+y*width]].z;
		merged_cluster_label[x+y*width] = ref[merged_cluster_label[x+y*width]];


	}
	
	//if(merged_cluster_label[x+y*width] == cluster_label[x+y*width]){
	//	//label...¡‚ÌêŠ  //current...‚»‚ÌêŠ‚Ìlabel
	//	int current = ref[cluster_label[x+y*width]];
	//	//‚»‚ÌêŠ‚Ìlabel‚ª‚Â‚¢‚½—Ìˆæ‚ÅêŠ‚Ì’l(x+y*width)‚Ælabel‚ªˆê’v‚·‚éêŠ‚ð’Tõ
	//	do{
	//		current = ref[current];
	//	} while(current != ref[current]);
	//	//’Tõ‚µ‚½label(‚Â‚Ü‚è‚»‚Ìlabel”Ô†‚ª‚Â‚¢‚½êŠ‚Ì’l)
	//	ref[cluster_label[x+y*width]] = current;		
	//}
	////Labeling phase
	//if(merged_cluster_label[x+y*width]> -1)
	//	//if(cluster_label[x+y*width] != cluster_label[ref[merged_cluster_label[x+y*width]]])
	//	merged_cluster_label[x+y*width] = ref[merged_cluster_label[x+y*width]];


}

__device__ inline void fatomicAdd(float* address, float val)
{
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_int, 
						assumed, 
						__float_as_int(val + __int_as_float(assumed)));
	
	}while(assumed != old);
}
__device__ inline void atomicFloatAdd(float *address, float val)
{
    int i_val = __float_as_int(val);
    int tmp0 = 0;
    int tmp1;
 
    while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
    {
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}
__global__ void countKernel(int* merged_cluster_label,
							float4* input_nd,
							int* cluster_label,
							float3* cluster_center,
							float3* sum_of_merged_cluster_normals, 
							float3* sum_of_merged_cluster_centers,
							int* merged_cluster_size, int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	if(merged_cluster_label[x + y * width]> -1 && (input_nd[x + y * width].x != -1.0f ||
														input_nd[x + y * width].y != -1.0f ||
															input_nd[x + y * width].y != -1.0f )){

		////count cluster size
		atomicAdd(&merged_cluster_size[merged_cluster_label[x + y * width]], 1);
		////add cluster normal
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x, input_nd[x + y * width].x);
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y, input_nd[x + y * width].y);
		atomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z, input_nd[x + y * width].z);
		//add cluster center
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x, cluster_center[cluster_label[x + y * width]].x);
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y, cluster_center[cluster_label[x + y * width]].y);
		atomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z, cluster_center[cluster_label[x + y * width]].z);
		//atomicFloatAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x, input_nd[x + y * width].x);
		//atomicFloatAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y, input_nd[x + y * width].y);
		//atomicFloatAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z, input_nd[x + y * width].z);
		////add cluster center
		//atomicFloatAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x, cluster_center[x + y * width].x);
		//atomicFloatAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y, cluster_center[x + y * width].y);
		//atomicFloatAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z, cluster_center[x + y * width].z);
		//fatomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x, input_nd[x + y * width].x);
		//fatomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y, input_nd[x + y * width].y);
		//fatomicAdd(&sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z, input_nd[x + y * width].z);
		////add cluster center
		//fatomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x, cluster_center[x + y * width].x);
		//fatomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y, cluster_center[x + y * width].y);
		//fatomicAdd(&sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z, cluster_center[x + y * width].z);
		
	}
	else{
		merged_cluster_label[x + y * width] = -1;
		//sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x = 0.0f;
		//sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y = 5.0f;
		//sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z = 5.0f;
		//sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x = 0.0f;
		//sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y = 0.0f;
		//sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z = 0.0f;
	}
	
}

__global__ void calculate_nd(int* merged_cluster_label,
							int* ref,
							float4* merged_cluster_nd,
							float3* sum_of_merged_cluster_normals, 
							float3* sum_of_merged_cluster_centers,
							int* merged_cluster_size, 
							float* merged_cluster_variance, 
							float4* input_nd,
							int sp_size, 
							int width, int height){
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	//float3 sum_of_normals;
	//sum_of_normals.x = sum_of_merged_cluster_normals[nlabel[x + y * width].label].x;
	//sum_of_normals.y = sum_of_merged_cluster_normals[nlabel[x + y * width].label].y;
	//sum_of_normals.z = sum_of_merged_cluster_normals[nlabel[x + y * width].label].z;
	if(merged_cluster_label[x + y * width] > -1 ){
		//calculate normal
		////merged_cluster_nd[x + y * width].x = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		////merged_cluster_nd[x + y * width].y = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		////merged_cluster_nd[x + y * width].z = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//merged_cluster_nd[merged_cluster_label[x + y * width]].x = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x;
		//merged_cluster_nd[merged_cluster_label[x + y * width]].y = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y;
		//merged_cluster_nd[merged_cluster_label[x + y * width]].z = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z;
		//float norm = sqrtf(merged_cluster_nd[merged_cluster_label[x + y * width]].x*merged_cluster_nd[merged_cluster_label[x + y * width]].x +
		//						merged_cluster_nd[merged_cluster_label[x + y * width]].y*merged_cluster_nd[merged_cluster_label[x + y * width]].y +
		//							merged_cluster_nd[merged_cluster_label[x + y * width]].z*merged_cluster_nd[merged_cluster_label[x + y * width]].z);
		//merged_cluster_nd[merged_cluster_label[x + y * width]].x /= norm;
		//merged_cluster_nd[merged_cluster_label[x + y * width]].y /= norm;
		//merged_cluster_nd[merged_cluster_label[x + y * width]].z /= norm;
		//
		////calculate center
		//float3 center;
		//center.x = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//center.y = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//center.z = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		////calculate distance
		//merged_cluster_nd[merged_cluster_label[x + y * width]].w = fabs(merged_cluster_nd[merged_cluster_label[x + y * width]].x*center.x + 
		//																	merged_cluster_nd[merged_cluster_label[x + y * width]].y*center.y +
		//																		merged_cluster_nd[merged_cluster_label[x + y * width]].z*center.z);
		
		
		//calculate normal
		merged_cluster_nd[x + y * width].x = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		merged_cluster_nd[x + y * width].y = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		merged_cluster_nd[x + y * width].z = sum_of_merged_cluster_normals[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate center
		float3 center;
		center.x = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].x/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		center.y = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].y/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		center.z = sum_of_merged_cluster_centers[merged_cluster_label[x + y * width]].z/(float)merged_cluster_size[merged_cluster_label[x + y * width]];
		//calculate variance
		float variance = input_nd[x + y * width].x*merged_cluster_nd[x + y * width].x +
								input_nd[x + y * width].y*merged_cluster_nd[x + y * width].y +
									input_nd[x + y * width].z*merged_cluster_nd[x + y * width].z;
		variance /= (float)merged_cluster_size[merged_cluster_label[x + y * width]];
		atomicAdd(&merged_cluster_variance[merged_cluster_label[x + y * width]], variance);
		//calculate distance
		merged_cluster_nd[x + y * width].w = fabs(merged_cluster_nd[x + y * width].x*center.x + 
														merged_cluster_nd[x + y * width].y*center.y +
															merged_cluster_nd[x + y * width].z*center.z);
	}
		//		acos(variance[merged_cluster_label[x + y * width]]) < (3.141592653f / 8.0f)){
		////calculate distance
		//merged_cluster_nd[x + y * width].w = fabs(merged_cluster_nd[x + y * width].x*center.x + 
		//																	merged_cluster_nd[x + y * width].y*center.y +
		//																		merged_cluster_nd[x + y * width].z*center.z);
		//}
		//else 
		//	merged_cluster_nd[x + y * width].w = -1.0f;
		//}
}

void LabelEquivalenceSeg::labelImage(float3* cluster_normals_device, int* cluster_label_device, float3* cluster_centers_device, float* variance_device, int sp_size){
	

	//initialize parameter
	initLabel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
		(InputND_Device, MergedClusterLabel_Device, ref, cluster_normals_device, cluster_label_device, cluster_centers_device, width, height);
	////init cluster_nd
	//float4 reset;
	//reset.x = 0.0;
	//reset.y = 0.0;
	//reset.z = 0.0;
	//reset.w = 0.0;
	//
	//thrust::fill(
	//	thrust::device_ptr<float4>(MergedClusterND_Device),
	//	thrust::device_ptr<float4>(MergedClusterND_Device) + width * height,
	//	reset);
	
	for(int i = 0; i < 20; i++){
		//scan(cluster_label_device);
		scanKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(InputND_Device, MergedClusterLabel_Device, ref, cluster_label_device, variance_device, width, height);
		//analysis(cluster_label_device);
		analysisKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, ref, cluster_label_device, width, height);	
	}
	//init merged_cluster_size
	int int_zero = 0;
	thrust::fill(
		thrust::device_ptr<int>(merged_cluster_size),
		thrust::device_ptr<int>(merged_cluster_size) + width * height,
		int_zero);
	//init normap map, center map
	float3 float3_zero;
	float3_zero.x = 0.0f;
	float3_zero.y = 0.0f;
	float3_zero.z = 0.0f;
	thrust::fill(
		thrust::device_ptr<float3>(sum_of_merged_cluster_normals),
		thrust::device_ptr<float3>(sum_of_merged_cluster_normals) + width * height,
		float3_zero);
	thrust::fill(
		thrust::device_ptr<float3>(sum_of_merged_cluster_centers),
		thrust::device_ptr<float3>(sum_of_merged_cluster_centers) + width * height,
		float3_zero);
	//init variance map
	float float_zero = 0.0f;
	thrust::fill(
		thrust::device_ptr<float>(MergedClusterVariance_Device),
		thrust::device_ptr<float>(MergedClusterVariance_Device) + width * height,
		float_zero);
	//calculate each cluster parametor
	//count cluster size
	countKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, InputND_Device, cluster_label_device, cluster_centers_device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, width, height);
	//calculate normal map, plane distance and variance map
	calculate_nd<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
			(MergedClusterLabel_Device, ref, MergedClusterND_Device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, MergedClusterVariance_Device, InputND_Device, sp_size, width, height);
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////initialize parameter
	//initLabel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//	(MergedClusterND_Device, MergedClusterLabel_Device, ref, cluster_normals_device, cluster_label_device, cluster_centers_device, width, height);
	//////init cluster_nd
	////float4 reset;
	////reset.x = 0.0;
	////reset.y = 0.0;
	////reset.z = 0.0;
	////reset.w = 0.0;
	////
	////thrust::fill(
	////	thrust::device_ptr<float4>(MergedClusterND_Device),
	////	thrust::device_ptr<float4>(MergedClusterND_Device) + width * height,
	////	reset);
	//for(int i = 0; i < 20; i++){
	//	//scan(cluster_label_device);
	//	scanKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(InputND_Device, MergedClusterLabel_Device, ref, cluster_label_device, variance_device, width, height);
	//	//analysis(cluster_label_device);
	//	analysisKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(MergedClusterLabel_Device, ref, cluster_label_device, width, height);
	//
	////init merged_cluster_size
	//int int_zero = 0;
	//thrust::fill(
	//	thrust::device_ptr<int>(merged_cluster_size),
	//	thrust::device_ptr<int>(merged_cluster_size) + width * height,
	//	int_zero);
	////init normap map and center map
	//float3 float3_zero;
	//float3_zero.x = 0.0f;
	//float3_zero.y = 0.0f;
	//float3_zero.z = 0.0f;
	//thrust::fill(
	//	thrust::device_ptr<float3>(sum_of_merged_cluster_normals),
	//	thrust::device_ptr<float3>(sum_of_merged_cluster_normals) + width * height,
	//	float3_zero);
	//thrust::fill(
	//	thrust::device_ptr<float3>(sum_of_merged_cluster_centers),
	//	thrust::device_ptr<float3>(sum_of_merged_cluster_centers) + width * height,
	//	float3_zero);
	//
	////calculate each cluster parametor
	////postProcess(cluster_label_device, cluster_centers_device);
	////count cluster size
	//countKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(MergedClusterLabel_Device, InputND_Device, cluster_label_device, cluster_centers_device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, width, height);
	////calculate normal map and plane distance
	//calculate_nd<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
	//		(MergedClusterLabel_Device, ref, MergedClusterND_Device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, variance_device, sp_size, width, height);
	//
	//}
	//memcpy
	cudaMemcpy(MergedClusterLabel_Host, MergedClusterLabel_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(MergedClusterND_Host, MergedClusterND_Device, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);
	cudaMemcpy(ref_host, ref, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(MergedClusterVariance_Host, MergedClusterVariance_Device, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
	//for(int y=0; y<height; y++){
	//	for(int x=0; x<width; x++){
	//		std::cout << MergedClusterVariance_Host[y*width+x]<<std::endl;
	//	}
	//}
	//for(int y=0; y<height; y++){
	//	for(int x=0; x<width; x++){
	//		std::cout << "label: "<<MergedClusterLabel_Host[y*width+x]<<std::endl;

	//		std::cout << "x: "<<MergedClusterND_Host[y*width+x].x<<std::endl;
	//		std::cout << "y: "<<MergedClusterND_Host[y*width+x].y<<std::endl;
	//		std::cout << "z: "<<MergedClusterND_Host[y*width+x].z<<std::endl;
	//		std::cout << "w: "<<MergedClusterND_Host[y*width+x].w<<std::endl;
	//	}
	//}
}

//void LabelEquivalenceSeg::labelImage(int rows, int cols, float3* cluster_normals_device, int* cluster_label_device, 
//										float3* cluster_centers_device, SuperpixelSegmentation::superpixel* sp_data_device){
//
//	int init = 999999;
//	thrust::fill(
//	thrust::device_ptr<int>(MergedClusterLabel_Device),
//	thrust::device_ptr<int>(MergedClusterLabel_Device) + width * height,
//	init);
//	thrust::fill(
//	thrust::device_ptr<int>(ref),
//	thrust::device_ptr<int>(ref) + width * height,
//	init);
//	
//	//initialize parameter
//	initLabel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//		(InputND_Device, MergedClusterLabel_Device, ref, spColor_Device,  
//			cluster_normals_device, cluster_label_device, cluster_centers_device, sp_data_device, width, height);
//	//initLabel<32*32><<<dim3(cols, rows), dim3(32, 32)>>>
//	//	(InputND_Device, MergedClusterLabel_Device, ref, spColor_Device,  
//	//		cluster_normals_device, cluster_label_device, cluster_centers_device, sp_data_device, rows, cols, width, height);
//	//cudaMemcpy(MergedClusterLabel_Host, MergedClusterLabel_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
//	////cudaMemcpy(MergedClusterND_Host, MergedClusterND_Device, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);
//	//
//	//for(int i = 0; i < height; i++){
//    //            for(int j = 0; j < width; j++){
//	//			//if(MergedClusterLabel_Host[i*width+j] > -1 && i*width+j == ref_host[MergedClusterLabel_Host[i*width+j]]){
//	//				std::cout << "label: "<<MergedClusterLabel_Host[i*width+j]<<std::endl;
//	//				//std::cout << "nd: "<<MergedClusterND_Host[MergedClusterLabel_Host[i*width+j]].x<< ", "<<MergedClusterND_Host[i*width+j].y<<", "<<MergedClusterND_Host[i*width+j].z<<", "<<MergedClusterND_Host[i*width+j].w<<std::endl;
//	//				//}
//	//			}
//	//}
//	// cv::Vec3b* pixel;
//    //    unsigned char* color;
//    //    for(int i = 0; i < height; i++){
//    //            for(int j = 0; j < width; j++){
//    //                    pixel = &show.at<cv::Vec3b>(cv::Point2d(j, i));
//    //                    if(MergedClusterLabel_Host[j + i*width] > -1){
//	//							//std::cout << "nd: "<<MergedClusterND_Host[i*width+j].x<< ", "<<MergedClusterND_Host[i*width+j].y<<", "<<MergedClusterND_Host[i*width+j].z<<std::endl;
//    //                            color = color_pool[MergedClusterLabel_Host[j + i * width]];
//    //                            pixel->val[0] = color[0];
//    //                            pixel->val[1] = color[1];
//    //                            pixel->val[2] = color[2];
//    //                    } else {
//    //                            pixel->val[0] = 0;
//    //                            pixel->val[1] = 0;
//    //                            pixel->val[2] = 0;
//    //                    }
//    //            }
//    //    }
//	//	Writer << show;
//    //    cv::imshow("Labeled surfaces", show);
//	//	cv::waitKey(0);
//	//unsigned start;
//	//start = clock();
//	for(int i = 0; i < 20; i++){
//		//scan(cluster_label_device);
//		scanKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//			(InputND_Device, MergedClusterLabel_Device, spColor_Device, ref, cluster_label_device, width, height);
//		//analysis(cluster_label_device);
//		analysisKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//			(MergedClusterLabel_Device, ref, cluster_label_device,/* cluster_normals_device, cluster_centers_device,*/ width, height);
//			
//	}
//	////double iteration_time = (double)((clock() - start)/1000.0);
//	////std::cout <<"iteration: " <<iteration_time <<std::endl;
//	////start = clock();
//	////init cluster_nd
//	//float4 reset;
//	//reset.x = 0.0;
//	//reset.y = 0.0;
//	//reset.z = 0.0;
//	//reset.w = 0.0;
//	//
//	//thrust::fill(
//	//	thrust::device_ptr<float4>(MergedClusterND_Device),
//	//	thrust::device_ptr<float4>(MergedClusterND_Device) + width * height,
//	//	reset);
//	//
//	////init merged_cluster_size
//	//int int_zero = 0;
//	//thrust::fill(
//	//	thrust::device_ptr<int>(merged_cluster_size),
//	//	thrust::device_ptr<int>(merged_cluster_size) + width * height,
//	//	int_zero);
//	////init normap map and center map
//	//float3 float3_zero;
//	//float3_zero.x = 0.0f;
//	//float3_zero.y = 0.0f;
//	//float3_zero.z = 0.0f;
//	//thrust::fill(
//	//	thrust::device_ptr<float3>(sum_of_merged_cluster_normals),
//	//	thrust::device_ptr<float3>(sum_of_merged_cluster_normals) + width * height,
//	//	float3_zero);
//	//thrust::fill(
//	//	thrust::device_ptr<float3>(sum_of_merged_cluster_centers),
//	//	thrust::device_ptr<float3>(sum_of_merged_cluster_centers) + width * height,
//	//	float3_zero);
//	////calculate each cluster parametor
//	postProcess(cluster_label_device, cluster_centers_device);
//	////count cluster size
//	////countKernel<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//	////		(MergedClusterLabel_Device, InputND_Device, cluster_label_device, cluster_center_device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, width, height);
//	////calculate_nd<<<dim3(width / 32, height / 24), dim3(32, 24)>>>
//	////		(MergedClusterLabel_Device, ref, MergedClusterND_Device, sum_of_merged_cluster_normals, sum_of_merged_cluster_centers, merged_cluster_size, width, height);
//	////memcpy
//	cudaMemcpy(MergedClusterLabel_Host, MergedClusterLabel_Device, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
//	cudaMemcpy(MergedClusterND_Host, MergedClusterND_Device, sizeof(float4)*width*height, cudaMemcpyDeviceToHost);
//	//cudaMemcpy(ref_host, ref, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
//	//double post_time = (double)((clock() - start)/1000.0);
//	//std::cout <<"postprocess: " <<post_time <<std::endl;
//	for(int y=0; y<height; y++){
//		for(int x=0; x<width; x++){
//			std::cout << "label: "<<MergedClusterLabel_Host[y*width+x]<<std::endl;
//			std::cout << "x: "<<MergedClusterND_Host[y*width+x].x<<std::endl;
//			std::cout << "y: "<<MergedClusterND_Host[y*width+x].y<<std::endl;
//			std::cout << "z: "<<MergedClusterND_Host[y*width+x].z<<std::endl;
//			std::cout << "w: "<<MergedClusterND_Host[y*width+x].w<<std::endl;
//		}
//	}
//
//}
