include "mlir/Pass/PassBase.td"

def ReverseCx : Pass<"ReverseCx"> {
  let summary = "Decomposition pass that reverses the control and targets of "
                "each two-qubits Cx gate in a circuit";
  let constructor = "mqss::opt::createReverseCxPass()";
}

def CxToLowerHCzH : Pass<"CxToLowerHCzH"> {
  let summary = "Decomposition pass of Cx by H, Cz, and H";
  let constructor = "mqss::opt::createCxToLowerHCzHPass()";
}

def CxToUpperHCzH : Pass<"CxToUpperHCzH"> {
  let summary = "Decomposition pass of Cx by H, Cz, and H";
  let constructor = "mqss::opt::createCxToUpperHCzHPass()";
}

def CzToLowerHCxH : Pass<"CzToLowerHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "mqss::opt::createCzToLowerHCxHPass()";
}

def CzToUpperHCxH : Pass<"CzToUpperHCxH"> {
  let summary = "Decomposition pass of Cz by H, Cx, and H";
  let constructor = "mqss::opt::createCzToUpperHCxHPass()";
}

def RxToHRzH : Pass<"RxToHRzH"> {
  let summary = "Decomposition pass of Rx by H, Rz and H";
  let constructor = "mqss::opt::createRxToHRzHPass()";
}

def RzToHRxH : Pass<"RzToHRxH"> {
  let summary = "Decomposition pass of Rz by H, Rx, and H";
  let constructor = "mqss::opt::createRzToHRxHPass()";
}


def CrxToHCrzH : Pass<"CrxToHCrzH"> {
  let summary = "Decomposition pass of Crx by H, Crz and H";
  let constructor = "mqss::opt::createCrxToHCrzHPass()";
}

def CrzToHCrxH : Pass<"CrzToHCrxH"> {
  let summary = "Decomposition pass of Crz by H, Crx, and H";
  let constructor = "mqss::opt::createCrzToHCrxHPass()";
}
def SdgToSSS : Pass<"SdgToSSS"> {
  let summary = "Decomposition pass that replaces a Sdg gate by S S S gates";
  let constructor = "mqss::opt::createSdgToSSSPass()";
}

def SToSdgSdgSdg : Pass<"SToSdgSdgSdg"> {
  let summary = "Decomposition pass that replaces an S gate by Sdg Sdg Sdg";
  let constructor = "mqss::opt::createSToSdgSdgSdgPass()";
}

def SToTT : Pass<"SToTT"> {
  let summary = "Decomposition pass that replaces S by T T gates";
  let constructor = "mqss::opt::createSToTTPass()";
}

def XToHZH : Pass<"XToHZH"> {
  let summary = "Decomposition pass of X by H, Z and H";
  let constructor = "mqss::opt::createXToHZHPass()";
}

def ZToHXH : Pass<"ZToHXH"> {
  let summary = "Decomposition pass of Z by H, X and H";
  let constructor = "mqss::opt::createZToHXHPass()";
}

def SwapToLowerCxCxCx : Pass<"SwapToLowerCxCxCx"> {
  let summary = "Decomposition pass of Swap by Cx Cx Cx gates";
  let constructor = "mqss::opt::createSwapToLowerCxCxCxPass()";
}

def SwapToUpperCxCxCx : Pass<"SwapToUpperCxCxCx"> {
  let summary = "Decomposition pass of Swap by Cx Cx Cx gates";
  let constructor = "mqss::opt::createSwapToUpperCxCxCxPass()";
}
def HToRzXRz: Pass <"HToRzXRz"> {
let summary = "Decomposition pass of H To Rz(pi) X Rz(pi/2) ";
let constructor = "mqss::opt::createHToRzXRzPass()";
}


def XToRx : Pass<"XToRx"> {
  let summary = "Decomposition pass of X into Rx rotation";
  let constructor = "mqss::opt::createXToRxPass()";
}

def YToRy : Pass<"YToRy"> {
  let summary = "Decomposition pass of Y into Ry rotation";
  let constructor = "mqss::opt::createYToRyPass()";
}

def ZToRz : Pass<"ZToRz"> {
  let summary = "Decomposition pass of Z into Rz rotation";
  let constructor = "mqss::opt::createZToRzPass()";
}

def SToRz : Pass<"SToRz"> {
  let summary = "Decomposition pass of S into Rz rotation";
  let constructor = "mqss::opt::createSToRzPass()";
}

def SdgToRz : Pass<"SdgToRz"> {
  let summary = "Decomposition pass of Sdg into Rz rotation";
  let constructor = "mqss::opt::createSdgToRzPass()";
}

def TToRz : Pass<"TToRz"> {
  let summary = "Decomposition pass of T into Rz rotation";
  let constructor = "mqss::opt::createTToRzPass()";
}

def R1ToRz : Pass<"R1ToRz"> {
  let summary = "Decomposition pass of R1 into Rz rotation";
  let constructor = "mqss::opt::createR1ToRzPass()";
}

def U2ToRzRyRz : Pass<"U2ToRzRyRz"> {
  let summary = "Decomposition pass of U2 into Rz-Ry-Rz sequence";
  let constructor = "mqss::opt::createU2ToRzRyRzPass()";
}

def U3ToRzRyRz : Pass<"U3ToRzRyRz"> {
  let summary = "Decomposition pass of U3 into Rz-Ry-Rz sequence";
  let constructor = "mqss::opt::createU3ToRzRyRzPass()";
}

def RyToRzRyRz : Pass<"RyToRzRyRz"> {
  let summary = "Decomposition pass of Ry into Rz-Ry-Rz sequence";
  let constructor = "mqss::opt::createRyToRzRyRzPass()";
}

def CyToSCxSdg : Pass<"CyToSCxSdg"> {
  let summary = "Decomposition pass of CY into S-CX-Sdg sequence";
  let constructor = "mqss::opt::createCyToSCxSdgPass()";
}

def RzToU3 : Pass<"RzToU3"> {
  let summary = "Decomposition pass of Rz into U3 gate";
  let constructor = "mqss::opt::createRzToU3Pass()";
}

def CRxToCRz : Pass<"CRxToCRz"> {
  let summary = "Decomposition pass of CRx into CRz rotations";
  let constructor = "mqss::opt::createCRxToCRzPass()";
}

def CRyToRzCRxRz : Pass<"CRyToRzCRxRz"> {
  let summary = "Decomposition pass of CRy into Rz-CRx-Rz sequence";
  let constructor = "mqss::opt::createCRyToRzCRxRzPass()";
}

def CRzToRzCxRzCx : Pass<"CRzToRzCxRzCx"> {
  let summary = "Decomposition pass of CRz into Rz-CX-Rz-CX sequence";
  let constructor = "mqss::opt::createCRzToRzCxRzCxPass()";
}
