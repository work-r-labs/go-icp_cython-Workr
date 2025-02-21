from __future__ import annotations
from typing import overload, Any, List, Dict, Tuple, Set, Sequence, Union

from enum import Enum as _PyEnum




class GoICP:
    """
    Cython implementation of _GoICP
    """
    
    Nm: int
    
    Nd: int
    
    MSEThresh: float
    
    SSEThresh: float
    
    icpThresh: float
    
    trimFraction: float
    
    inlierNum: int
    
    doTrim: bool
    
    def __init__(self) -> None:
        """
        Cython signature: void GoICP()
        """
        ...
    
    def Register(self) -> float:
        """
        Cython signature: float Register()
        """
        ...
    
    def BuildDT(self) -> None:
        """
        Cython signature: void BuildDT()
        """
        ...
    
    def optimalRotation(self) -> List[List[float]]:
        """
        Cython signature: libcpp_vector[libcpp_vector[double]] optimalRotation()
        """
        ...
    
    def optimalTranslation(self) -> List[float]:
        """
        Cython signature: libcpp_vector[double] optimalTranslation()
        """
        ...
    
    def loadModelAndData(self, in_0: int , in_1: List[POINT3D] , in_2: int , in_3: List[POINT3D] ) -> None:
        """
        Cython signature: void loadModelAndData(int, libcpp_vector[POINT3D], int, libcpp_vector[POINT3D])
        """
        ...
    
    def setInitNodeRot(self, in_0: ROTNODE ) -> None:
        """
        Cython signature: void setInitNodeRot(ROTNODE &)
        """
        ...
    
    def setInitNodeTrans(self, in_0: TRANSNODE ) -> None:
        """
        Cython signature: void setInitNodeTrans(TRANSNODE &)
        """
        ...
    
    def setDTSizeAndFactor(self, in_0: int , in_1: float ) -> None:
        """
        Cython signature: void setDTSizeAndFactor(int, double)
        """
        ... 


class POINT3D:
    """
    Cython implementation of _POINT3D
    """
    
    x: float
    
    y: float
    
    z: float
    
    @overload
    def __init__(self, ) -> None:
        """
        Cython signature: void POINT3D()
        """
        ...
    
    @overload
    def __init__(self, in_0: float , in_1: float , in_2: float ) -> None:
        """
        Cython signature: void POINT3D(float, float, float)
        """
        ...
    
    def pointToString(self) -> None:
        """
        Cython signature: void pointToString()
        """
        ... 


class ROTNODE:
    """
    Cython implementation of _ROTNODE
    """
    
    a: float
    
    b: float
    
    c: float
    
    w: float
    
    ub: float
    
    lb: float
    
    l: int
    
    def __init__(self) -> None:
        """
        Cython signature: void ROTNODE()
        """
        ... 


class TRANSNODE:
    """
    Cython implementation of _TRANSNODE
    """
    
    x: float
    
    y: float
    
    z: float
    
    w: float
    
    ub: float
    
    lb: float
    
    def __init__(self) -> None:
        """
        Cython signature: void TRANSNODE()
        """
        ... 

