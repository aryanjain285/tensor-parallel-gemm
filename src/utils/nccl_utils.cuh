// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University
//
// nccl_utils.cuh - NCCL error checking macro.

#pragma once

#include <nccl.h>

#include <cstdio>
#include <cstdlib>

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
