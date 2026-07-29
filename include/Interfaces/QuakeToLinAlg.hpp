
/** @file
 * @brief
 * @details This header defines a set of functions utilized to convert Quake
 * circuits to Arith + LinAlg
 *
 * @par
 * This header file is used by the QASM3ToQuakePass to perform the conversion of
 * QASM programs to MLIR/Quake modules.
 */

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using func::FuncOp;

namespace mqss::interfaces {
    /**
     * @brief Given a module and a quake quantum kernel. This method perform the
     transformation from Quake to LinAlg + Arith.
       @details This method converts any given Quake quantum kernel to a sequence of
     complex vectors/matrices multiplications corresponding to the input quantum
     kernel. This representation might be useful, later as input of IREE compiler
     which is able to generate GPU code for many vendors.
        @param[out] module is the mlir module where my input quantum kernel is. In
     module the converted function will be inserted.
        @param[in] quakeFunction is the quantum kernel to be converted.
        @param[in] builder is an `OpBuilder` object associated with a MLIR module.
     It is used to insert new instructions to the corresponding MLIR module.
        @param gpuFunction
        @param[in] tensorType is the datatype associated to the state vector
        @param[in] matrixType is the datatype associated to the gate matrices
        @param[in] numberOfQubits is the number of qubits utilized by the quantum
     kernel
    */
    Value convertQuakeToLinAlg(
        ModuleOp module, FuncOp quakeFunction, OpBuilder &builder,
        FuncOp gpuFunction, RankedTensorType tensorType,
        RankedTensorType matrixType, int numberOfQubits);
} // namespace mqss::interfaces
