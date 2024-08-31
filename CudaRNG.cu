#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <cuda_runtime.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <errno.h>


#define Cm cudaMalloc
#define Cmc cudaMemcpy
#define CmcHD cudaMemcpyHostToDevice
#define CmcDH cudaMemcpyDeviceToHost


__host__ void cpuBinConverter(int temprature,int* fin_bin)
{
    for(int i=31;i>=0;i--)
    {
        int x=((int)temprature>>i)&1;
        fin_bin[31-i]=x;
    }
}

__global__ void gpuBinConverter(int d_temprature,int* d_fin_bin)
{
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ int hold;
    if(tid<32)
    {
        int hold=((int)d_temprature>>tid)&1;
        d_fin_bin[31-tid]=hold;
    }
}

__host__ void cpuIntConverter(int* cells,unsigned int &ret,int size)
{
    int hold;
    for(int i=0;i<size;i++)
    {
        hold+=pow(2,size-i-1)*cells[i];
    }
    ret=hold;
}

__global__ void gpuIntConverter(int* d_cells,unsigned int *d_ret,int d_size)
{
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ int hold;
    if(tid==0)
    {
        hold=0;
    }
    if(tid<d_size)
    {
        atomicAdd(&hold,pow(2,d_size-tid-1)*d_cells[tid]);
    }
    __syncthreads();
    if(tid==0)
    {
        *d_ret=hold;

    }
}

__global__ void moveRight(int* arr,int* res,int delta)
{
    int value=arr[threadIdx.x];
    value=__shfl_up_sync(0xFFFFFFFF,value,delta);
    res[threadIdx.x]=value;
    res[0]=0;
}

__global__ void hillisSteeleScan(int* arr,int *temp,int size)
{

    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    temp[tid]=arr[tid];
    __syncthreads();
    for(int stride=1;stride<=(size)/2;stride*=2)
    {
        if(tid>=stride)
        {
            temp[tid]=arr[tid]+arr[tid-stride];
        }
        else
        {
            temp[tid]=arr[tid];
        }
        __syncthreads();

        arr[tid]=temp[tid];

        __syncthreads();
    }
}   

__host__ void cpuTransf(int* cells,int* out, int size)
{
    for(int i=0;i<size;i++)
    {
        out[i]=cells[i];
    }
}

__global__ void gpuTransf(int* d_cells,int* d_out,int d_size)
{
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    if(tid<d_size)
    {
        atomicAdd(&d_out[tid],d_cells[tid]);
    }
    __syncthreads();
}

__host__ void elementaryCellGenerate(int* cells,int* out,int size,int* rule)
{
    int p,q,r;
    for(int i=1;i<size-1;i++)
    {
        p=4*cells[i-1];
        q=2*cells[i];
        r=cells[i+1];
        out[i]=rule[7-(p+q+r)];
    }
}

__global__ void gpuElementaryCellularAutoRule30(int* d_cells,int* d_out, int d_size,int* d_rule)
{
    int tid=threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ int p,q,r;
    __shared__ int hold;
    if(tid>0 && tid<d_size-1)
    {
        p=4*d_cells[tid-1];
        q=2*d_cells[tid];
        r=d_cells[tid+1];
        hold=d_rule[7-(p+q+r)];
        d_out[tid]=hold;
    }
}   

__host__ void cpuPrinter(int* cells,int size)
{
    for(int i=0;i<size;i++)
    {
        if(cells[i]==1) printf("1");
        else printf("0");
    }
    printf("\n");
}

__global__ void gpuPrinter(int *d_cells,int d_size)
{
    for(int i=0;i<d_size;i++)
    {
        if(d_cells[i]==1) printf("1");
        else printf("0");
    }
    printf("\n");
}

__host__ void cudaRNG(int seed,int size,int* rule,int blocksize)
{
    int *cells=(int*)malloc(size*sizeof(int));
    unsigned int ret;
    unsigned int *d_ret;
    int *d_cells;
    int *d_res;
    int *d_out;
    int *d_rule;

    cpuBinConverter(seed,cells);
    cpuPrinter(cells,size);

    Cm((int**)&d_cells,size*sizeof(int));
    Cm((int**)&d_res,size*sizeof(int));//temporary variables
    Cm((int**)&d_out,size*sizeof(int));//temporary variables 
    Cm((void**)&d_ret,sizeof(unsigned int));
    Cm((int**)&d_rule,8*sizeof(int));

    Cmc(d_cells,cells,size*sizeof(int),CmcHD);
    Cmc(d_rule,rule,8*sizeof(int),CmcHD);

    int no_blocks=(size+blocksize-1)/blocksize;
    for(int i=0;i<(size/2);i++)
    {
        gpuPrinter<<<1,1>>>(d_cells,size);
        gpuElementaryCellularAutoRule30<<<no_blocks,blocksize>>>(d_cells,d_out,size,d_rule);

        cudaDeviceSynchronize();

        gpuTransf<<<no_blocks,blocksize>>>(d_out,d_cells,size);
    }
    gpuIntConverter<<<1,32>>>(d_cells,d_ret,size);

    cudaDeviceSynchronize();

    Cmc(&ret,d_ret,sizeof(unsigned int),CmcDH);
    printf("%d",ret);
}

int main(int argc, char** argv) 
{
    char path[256];
    char filePath[512];
    DIR *dir;
    FILE *file;
    float temperature;
    float fin_temp;
    struct dirent *ent;
    
    strcpy(path,"/sys/class/hwmon/");

    dir=opendir(path);
    if(dir==NULL)
    {
        perror("opendir");
        return 1;
    }

    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        snprintf(filePath,sizeof(filePath),"%s%s",path,ent->d_name);

        DIR *subdir = opendir(filePath);
        if (subdir==NULL) 
        {
            perror("opendir subdir");
            continue;
        }

        struct dirent *subent;
        while ((subent = readdir(subdir)) != NULL) {
            if (strstr(subent->d_name, "temp") && strstr(subent->d_name, "_input")) {
                snprintf(filePath,sizeof(filePath),"%s/%s/%s",path,ent->d_name,subent->d_name);

                file = fopen(filePath,"r");
                if (file==NULL) 
                {
                    perror("fopen");
                    continue;
                }

                fscanf(file, "%f", &temperature);
                printf("%2f \n",temperature/1000.0);
                fin_temp=temperature/1000.0;
                fclose(file);
            }
        }
        closedir(subdir);
    }
    closedir(dir);

    int rule[8]={0,0,0,1,1,1,1,0};
    int out[64]={0};
    unsigned int r;

    //gpu generation implemenatation
    printf("The GPU implementation: \n");
    cudaRNG((int)fin_temp,64,rule,32);
    printf("\n");

    int *d_temp;
    int *d_cells;
    
    //cpu generation for refernce
    printf("The CPU implementation: \n");
    // for(int i=0;i<32;i++)
    // {
    //     cpuPrinter(cells,64);
    //     elementaryCellGenerate(cells,out,64,rule);
    //     cpuTransf(out,cells,64);
    // }
    // cpuPrinter(out,64);
    // cpuIntConverter(out,r,64);
    // printf("The random number generated is %d", r);

    return 0;
}
