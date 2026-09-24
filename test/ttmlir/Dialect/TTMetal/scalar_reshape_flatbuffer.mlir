// RUN: ttmlir-opt --ttcore-register-device -o %t.mlir %s
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir

// Scalar reshapes in host wrappers use arithmetic constants as indices.
// Translation must register the arithmetic dialect to parse these wrappers.
module {
  func.func @scalar_reshape(%input: memref<f32>) -> memref<1x1xf32> attributes {tt.function_type = "forward_device"} {
    %zero = arith.constant 0 : index
    %output = memref.alloc() : memref<1x1xf32>
    %value = memref.load %input[] : memref<f32>
    memref.store %value, %output[%zero, %zero] : memref<1x1xf32>
    return %output : memref<1x1xf32>
  }
}
