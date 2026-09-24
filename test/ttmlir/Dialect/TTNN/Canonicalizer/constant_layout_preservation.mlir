// RUN: ttmlir-opt --ttcore-register-device --canonicalize %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize %s | FileCheck %s

#host = #ttnn.buffer_type<system_memory>
#dram = #ttnn.buffer_type<dram>
#host_rm = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2xf32, #host>>
#host_tile = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #host>>
#dram_rm = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2xf32, #dram>, <interleaved>>
#dram_tile = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// CHECK-DAG: #[[HOST:.*]] = #ttnn.ttnn_layout<{{.*}}memref<1x2xf32, #system_memory>>

// Non-splat constants must remain in row-major host memory.
// CHECK-LABEL: func.func @constant_to_dram_rm
// CHECK: %[[CONST:.*]] = "ttnn.constant"()
// CHECK-SAME: dense<[1.000000e+00, 2.000000e+00]>
// CHECK-SAME: -> tensor<2xf32, #[[HOST]]>
// CHECK: %[[CONVERT:.*]] = "ttnn.to_tensor_spec"(%[[CONST]])
// CHECK: return %[[CONVERT]]
func.func @constant_to_dram_rm() -> tensor<2xf32, #dram_rm> {
  %0 = "ttnn.constant"() <{value = dense<[1.0, 2.0]> : tensor<2xf32>}> : () -> tensor<2xf32, #host_rm>
  %1 = "ttnn.to_tensor_spec"(%0) : (tensor<2xf32, #host_rm>) -> tensor<2xf32, #dram_rm>
  return %1 : tensor<2xf32, #dram_rm>
}

// CHECK-LABEL: func.func @constant_to_dram_tile
// CHECK: %[[CONST:.*]] = "ttnn.constant"()
// CHECK-SAME: dense<[1.000000e+00, 2.000000e+00]>
// CHECK-SAME: -> tensor<2xf32, #[[HOST]]>
// CHECK: %[[CONVERT:.*]] = "ttnn.to_tensor_spec"(%[[CONST]])
// CHECK: return %[[CONVERT]]
func.func @constant_to_dram_tile() -> tensor<2xf32, #dram_tile> {
  %0 = "ttnn.constant"() <{value = dense<[1.0, 2.0]> : tensor<2xf32>}> : () -> tensor<2xf32, #host_rm>
  %1 = "ttnn.to_tensor_spec"(%0) : (tensor<2xf32, #host_rm>) -> tensor<2xf32, #dram_tile>
  return %1 : tensor<2xf32, #dram_tile>
}

// CHECK-LABEL: func.func @constant_to_host_tile
// CHECK: %[[CONST:.*]] = "ttnn.constant"()
// CHECK-SAME: dense<[1.000000e+00, 2.000000e+00]>
// CHECK-SAME: -> tensor<2xf32, #[[HOST]]>
// CHECK: %[[CONVERT:.*]] = "ttnn.to_tensor_spec"(%[[CONST]])
// CHECK: return %[[CONVERT]]
func.func @constant_to_host_tile() -> tensor<2xf32, #host_tile> {
  %0 = "ttnn.constant"() <{value = dense<[1.0, 2.0]> : tensor<2xf32>}> : () -> tensor<2xf32, #host_rm>
  %1 = "ttnn.to_tensor_spec"(%0) : (tensor<2xf32, #host_rm>) -> tensor<2xf32, #host_tile>
  return %1 : tensor<2xf32, #host_tile>
}

// Splat constants can become full ops, which support device creation.
// CHECK-LABEL: func.func @splat_to_device
// CHECK: "ttnn.full"
// CHECK-NOT: "ttnn.to_tensor_spec"
// CHECK: return
func.func @splat_to_device() -> tensor<2xf32, #dram_tile> {
  %0 = "ttnn.constant"() <{value = dense<1.0> : tensor<2xf32>}> : () -> tensor<2xf32, #host_rm>
  %1 = "ttnn.to_tensor_spec"(%0) : (tensor<2xf32, #host_rm>) -> tensor<2xf32, #dram_tile>
  return %1 : tensor<2xf32, #dram_tile>
}
