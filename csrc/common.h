#ifndef __COMMON_H__
#define __COMMON_H__
#include <set>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <torch/extension.h>


std::tuple<int, int> calculate_resolution(int ori_h, int ori_w, int kernel, int padding, int stride, int dilation)
{
    // Calculate the output resolution of the output tensor
    // printf("calculating output size: h:%d kernel:%d padding:%d stride:%d\n",ori_h, kernel, padding, stride);
    int h, w;
    h = int((ori_h + 2 * padding - kernel) / stride) + 1;
    w = int((ori_w + 2 * padding - kernel) / stride) + 1;
    return std::make_tuple(h, w);
}

inline int calculate_index(int id1, int shift1, int id2, int shift2, int id3, int shift3, int id4, int shift4 = 1)
{
    return id1 * shift1 + id2 * shift2 + id3 * shift3 + id4 * shift4;
}

bool verify_bcsr(int * mask, float * data, int h, int w, int block_h, int block_w, int* row, int * col, float* values)
{
    for(int rid=0; rid<h/block_h; rid++){
        // printf("row-%d: %d row-%d : %d\n", rid, row[rid], rid+1, row[rid+1]);
        int _start = row[rid];
        int _end = row[rid+1];
        for(int _pos=_start; _pos<_end; _pos++){
            int cid = col[_pos];
            for(int i=0;i<block_h;i++){
                for(int j=0;j<block_w;j++){
                    int offset = (rid * block_h+i) * w + cid * block_w + j;
                    int csr_offset = _pos * block_h * block_w + i * block_w + j;
                    if (mask[offset]>0){
                        // printf("%f %f\n", data[offset], values[csr_offset]);
                        if(abs(data[offset]-values[csr_offset])>1e-8)
                        {
                            return false;
                        }
                        mask[offset]= 0;
                    }
                }
            }
        }
    }
    printf("%d blocks remained\n", row[h/block_h]);
    for(int i=0;i<block_h*block_w;i++)
        if(mask[i])
            return false;
    return true;
}

#endif