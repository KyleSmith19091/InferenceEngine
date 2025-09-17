//go:build darwin

#include "metal.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

// Interface to the local GPU device
id<MTLDevice> device;

// Representations of compiled compute programs to execute on the GPU
id<MTLComputePipelineState> pipelineStateNaive;

// Used to create and submit command buffers to the GPU device
id<MTLCommandQueue> commandQueue;

// Buffers of input and output data being passed to and from the GPU
id<MTLBuffer> bufferA;
id<MTLBuffer> bufferB;
id<MTLBuffer> bufferC;

/**
 * Compiles and creates the Metal shader library used later on to execute commands on the GPU.
 * Initializes the pipeline state objects for the relevant public functions defined in the 
 * Metal shader code.
 */
void
initializePipelineAndCommandQueue (char *source_path, char* kernel_name) 
{
  device = MTLCreateSystemDefaultDevice();
  NSLog(@"Using default device %s", [device.name UTF8String]);

  NSError *error = nil;

  // Compile and initialize a new library located at the provided source path.
  MTLCompileOptions *compileOptions = [MTLCompileOptions new];
  compileOptions.languageVersion = MTLLanguageVersion2_4;
  NSString *ss = [NSString stringWithUTF8String:source_path];

  id<MTLLibrary> lib = [device newLibraryWithSource:ss
    options:compileOptions
    error:&error];

  if (lib == nil) {
    NSLog(@"Failed to create library, error %@.", error);
    return;
  }

  // Create a representation of the naive multiplication public shader function in 
  // the Metal library created above
  NSString *kernel = [NSString stringWithUTF8String:kernel_name];
  id<MTLFunction> naiveFunction =
      [lib newFunctionWithName:kernel];
  if (naiveFunction == nil) {
    NSLog(@"Failed to find the matrix_multiply_naive function.");
    return;
  }

  pipelineStateNaive = [device newComputePipelineStateWithFunction:naiveFunction
    error:&error];
  if (pipelineStateNaive == nil) {
    NSLog(@"Failed to create naive pipeline state object, error %@.", error);
    return;
  }


  commandQueue = [device newCommandQueue];
  if (commandQueue == nil) {
    NSLog(@"Failed to find the command queue.");
    return;
  }
}

/**
 * Initialize the two input buffers containing matrix data, and also prepare the output buffer
 * for the resulting matrix multiplication result.
 */
void 
initializeMTLBuffers (
  void* a, 
  void* b, 
  int data_size_bytes, 
  int a_array_size,
  int b_array_size,
  int out_array_size
) {
  bufferA = [device newBufferWithBytes:a 
    length:a_array_size*data_size_bytes 
    options:MTLResourceStorageModeShared];

  bufferB = [device newBufferWithBytes:b 
    length:b_array_size*data_size_bytes 
    options:MTLResourceStorageModeShared];

  // Resulting matrix buffer
  bufferC = [device newBufferWithLength:out_array_size*data_size_bytes 
    options:MTLResourceStorageModeShared];
}

/**
 * Configures GPU grids, serializes input parameters and buffers into the compute encoder, and executes the commands.
 * This assumes that the correct pipeline state has already been set to the naive
 * metal matrix multiplication kernel functions.
 */
void*
metal_mult (MatrixParams *params, id<MTLComputePipelineState> pipelineState)
{
    @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) {
      NSLog(@"Failed to get the command buffer.");
      return nil;
    }
    // Get the compute encoder.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) {
      NSLog(@"Failed to get the compute encoder.");
      return nil;
    }

    // Sets the context for the appropriate Metal kernel function
    [computeEncoder setComputePipelineState:pipelineState];

    // Indicates the dimensionality of the input matrix to the thread scheduler
    MTLSize threadsPerGrid = MTLSizeMake(params->a_cols, params->a_rows, 1);

    // Calculate a threadgroup size.
    // https://developer.apple.com/documentation/metal/calculating_threadgroup_and_grid_sizes?language=objc
    NSUInteger w = pipelineState.threadExecutionWidth;
    NSUInteger h = pipelineState.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

    [computeEncoder setBytes:params length:16 atIndex:0];
    [computeEncoder setBuffer:bufferA offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferB offset:0 atIndex:2];
    [computeEncoder setBuffer:bufferC offset:0 atIndex:3];

    // Encode the compute command.
    [computeEncoder dispatchThreads:threadsPerGrid 
      threadsPerThreadgroup:threadsPerThreadgroup];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    [commandBuffer commit];

    // We could add a completion event handler here instead and do other work, but since 
    // the matrix result is needed right away, we'll just block.
    // https://developer.apple.com/documentation/metal/mtlcommandbuffer/1442997-addcompletedhandler
    [commandBuffer waitUntilCompleted];

    return bufferC.contents;
  }
}

/**
 * Configures the GPU command encoder to call the naive matrix multiplication Metal kernel function.
 */
void*
metal_mult_naive (MatrixParams *params) 
{
  return metal_mult(params, pipelineStateNaive);
  // return mps_mult(params);
}

static void ensureDeviceAndQueue() {
  if (device == nil) {
    device = MTLCreateSystemDefaultDevice();
  }
  if (commandQueue == nil && device != nil) {
    commandQueue = [device newCommandQueue];
  }
}

void*
mtl_new_buffer(int length_bytes) {
  @autoreleasepool {
    ensureDeviceAndQueue();
    if (device == nil) {
      return nil;
    }
    id<MTLBuffer> buf = [device newBufferWithLength:length_bytes options:MTLResourceStorageModeShared];
    // using __bridge_retained to indicate that ownership should be transferred from being reference counted in objective-C to C-runtime where
    // it needs to be manually release
    // essentially transferring ownership to caller
    return (__bridge_retained void*)buf;
  }
}

void
mtl_release_buffer(void* buf) {
  if (buf == nil) return;
  id<MTLBuffer> o = (__bridge_transfer id<MTLBuffer>)buf;
  (void)o; // ARC will release
}

void
mtl_buffer_write(void* buf, void* src, int length_bytes) {
  if (buf == nil || src == nil || length_bytes <= 0) return;
  id<MTLBuffer> o = (__bridge id<MTLBuffer>)buf;
  memcpy(o.contents, src, (size_t)length_bytes);
}

void
mtl_buffer_read(void* buf, void* dst, int length_bytes) {
  if (buf == nil || dst == nil || length_bytes <= 0) return;
  id<MTLBuffer> o = (__bridge id<MTLBuffer>)buf;
  memcpy(dst, o.contents, (size_t)length_bytes);
}

void
mtl_buffer_read_at(void* buf, int offset_bytes, void* dst, int length_bytes) {
  if (buf == nil || dst == nil || length_bytes <= 0) return;
  id<MTLBuffer> o = (__bridge id<MTLBuffer>)buf;
  if (offset_bytes < 0) return;
  size_t off = (size_t)offset_bytes;
  size_t len = (size_t)length_bytes;
  if (off + len > o.length) return;
  memcpy(dst, (void *)((char*)o.contents + off), len);
}

static void*
metal_mult_with_buffers(MatrixParams *params, id<MTLComputePipelineState> pipelineState, id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c)
{
  @autoreleasepool {
    ensureDeviceAndQueue();
    if (device == nil || commandQueue == nil) return nil;
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) return nil;
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    if (computeEncoder == nil) return nil;
    [computeEncoder setComputePipelineState:pipelineState];
    MTLSize threadsPerGrid = MTLSizeMake(params->a_cols, params->a_rows, 1);
    NSUInteger w = pipelineState.threadExecutionWidth;
    NSUInteger h = pipelineState.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

    [computeEncoder setBytes:params length:16 atIndex:0];
    [computeEncoder setBuffer:a offset:0 atIndex:1];
    [computeEncoder setBuffer:b offset:0 atIndex:2];
    [computeEncoder setBuffer:c offset:0 atIndex:3];

    [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return c.contents;
  }
}

void*
metal_mult_naive_with_buffers(MatrixParams *params, void* bufA, void* bufB, void* bufC) {
  id<MTLBuffer> a = (__bridge id<MTLBuffer>)bufA;
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)bufB;
  id<MTLBuffer> c = (__bridge id<MTLBuffer>)bufC;
  return metal_mult_with_buffers(params, pipelineStateNaive, a, b, c);
}