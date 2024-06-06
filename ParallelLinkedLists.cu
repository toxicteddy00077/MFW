#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define ll long long
#define Cm cudaMalloc
#define Cmc cudaMemcpy
#define CmcHD cudaMemcpyHostToDevice
#define CmcDH cudaMemcpyDeviceToHost
#define SIZE sizeof(node)

typedef struct node
{
    ll data;
    struct node* next;
    struct node* prev;

    node(){ };
    __host__ void insertNode(node* n,ll data);
    __host__ void delNode(node*n);
}node;

//constructor
__host__ node* nodeConstructor(ll data)
{
    node* root=(node*)malloc(SIZE);
    if(root!=NULL)
    {
        root->data=data;
        root->next=NULL;
        root->prev=NULL;
    }
    return root;
}

typedef struct linkl
{
    int N;
    node* root;
    node** pos;
}linkl;

//linkedList constructor
__host__ linkl* linklConstructor(int N)
{
    linkl* l=(linkl*)malloc(sizeof(linkl));
    l->N=N;
    l->root=NULL;
    l->pos=(node**)malloc(N*sizeof(node*));

    return l;
}  

__host__ void addNode(linkl *l,ll data)
{
    node* temp=nodeConstructor(data);
    if(l->root==NULL)
    {
        l->root=temp;
    }
    else 
    {
        node* curr=l->root;
        while(curr->next!=NULL)
        {
            curr=curr->next;
        }
        curr->next=temp;
        temp->prev=curr;
    }
}

//Inserts node infront of node n (passed as parameter)
__host__ void node::insertNode(node* n,ll data)
{
    node* x=(node*)malloc(SIZE);
    x=nodeConstructor(data);
    if(n!=NULL)
    {
        x->next=n->next;
        n->next=x;
        x->prev=n;
        if(x->next!=NULL)
        {
            x->next->prev=x;
        }
    }
}

//deletes node n
__host__ void node::delNode(node *n)
{
    if (n->prev!=NULL) n->prev->next=n->next;
    if (n->next!=NULL) n->next->prev=n->prev;
    free(n);
}

__host__ void disp(linkl* l)
{
    node* curr=l->root;
    while(curr!=NULL)
    {
        printf("%lld ",curr->data);
        curr=curr->next;
    }
}

//returns node with searched data
__global__ void search(linkl*l,ll data)
{
    node* curr=l->root;
    int id=threadIdx.x+blockDim.x*blockIdx.x;
    if(id<l->N)
    {
        if(curr->data==data)
        {
            l->pos[id]=curr;
        }
        curr=curr->next;
        l->pos[id]=NULL;
    }
    return;
}   

__host__ node* searchRet(linkl* l,ll data)
{
    linkl* dl;
    node* d_root;
    node** d_pos;
    Cm((void**)&dl,sizeof(linkl));
    Cm((void**)&d_root,l->N*sizeof(node));
    Cm((void**)&d_pos,l->N*sizeof(node*));

    Cmc(dl,l,sizeof(linkl),CmcHD);
    Cmc(d_root,l->root,l->N*sizeof(node),CmcHD);
    Cmc(&(dl->root),&d_root,sizeof(node),CmcHD);
    Cmc(&(dl->pos),&d_pos,sizeof(node**),CmcHD);

    search<<<1,1>>>(dl,data);
    cudaDeviceSynchronize();

    Cmc(l->pos,d_pos,l->N*sizeof(node*),CmcDH);

    cudaFree(dl);
    cudaFree(d_pos);
    cudaFree(d_root);

    for(int i=0;i<l->N;i++)
    {
        if(l->pos[i]!=NULL)
        {
            continue;
        }
        else
        {
            node* temp=l->pos[i];
            l->pos[i]=NULL;
            return temp;
            break;
        }
    }
    return NULL;
}

//cycle detetction
__host__ void cycleDetect(linkl* l)
{
    node* curr=l->root;
    ll temp=l->root->data;
    while(curr!=NULL)
    {
        if(curr->data==temp)
        {
            printf("it is cycle");
            break;
        }
        curr=curr->next;
    }
}

//Hillis-Steele scan assign
__global__ void assign(linkl* l,int* hold)
{
    int* arr=(int*)malloc(sizeof(int));
    node* curr=l->root;
    for(int i=0;i<l->N;i++)
    {
        arr[i]=curr->data;
        curr=curr->next;
    }

    curr=l->root;

    int id=threadIdx.x+blockDim.x*blockIdx.x;
    if(id<l->N)
    {
        hold[id]=arr[id];
    }
    __syncthreads();
}           

__global__ void HSScan(linkl*l, int* hold)
{
    
}   

int main()
{
    linkl* l=linklConstructor(4);
    for(int i=1;i<=l->N;i++)
    {
        addNode(l,i);
    }

    search<<<1,1>>>(l,3);
    for(int i=0;i<l->N;i++)
    {
        if(l->pos[i]==NULL)
        {
            continue;
        }
        else 
        {
            printf("%d",l->pos[i]->data);
            break;
        }
    }
    return 0;
}