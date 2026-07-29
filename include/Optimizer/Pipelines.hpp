
/** @file
 * @brief
 * @details This header defines the three optimization levels supported by the
 * XXX.
 * @details This header defines the three optimization levels supported by the
 *  `O1`, `O2` and `O3`. Each function appends the corresponding list of
 * optimization passes to a given `mlir::PassManager` object.
 *
 * @par
 * This header must be included to use the different optimization levels that
 * are part of the XXX.
 */

#pragma once

#include "mlir/Pass/PassManager.h"

namespace mqss::opt {
/**
  @brief Function defining the optimization level `O1`.
  @details This functions appends to a `mlir::PassManager` the list of passes
  corresponding to optimization level `O1`.
  @param[out] pm is the `mlir::PassManager` after appending the list of passes
  corresponding to optimization level `O1`.
*/
void O1(mlir::PassManager &pm);
/**
  @brief Function defining the optimization level `O2`.
  @details This functions appends to a `mlir::PassManager` the list of passes
  corresponding to optimization level `O2`.
  @param[out] pm is the `mlir::PassManager` after appending the list of passes
  corresponding to optimization level `O2`.
*/
void O2(mlir::PassManager &pm);
/**
  @brief Function defining the optimization level `O3`.
  @details This functions appends to a `mlir::PassManager` the list of passes
  corresponding to optimization level `O3`.
  @param[out] pm is the `mlir::PassManager` after appending the list of passes
  corresponding to optimization level `O3`.
*/
void O3(mlir::PassManager &pm);
} // namespace mqss::opt
