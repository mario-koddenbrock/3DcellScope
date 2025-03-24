from enum import Enum
from dataclasses import dataclass
from typing import Union
class FILTERMSG(Enum):
    RESET=0
    EXCLUDE=1
    KEEP_ONLY=2


class PCATYPE(Enum):
    PCA=0
    LDA=1
class PCAMODE(Enum):
    AUTO=0
    SAMPLE=1
    DESCRIPTOR=2
    CLUSTER=3
    TARGET=4

    @staticmethod
    def from_str(mode:str):
        if mode[:4] == "Auto":return PCAMODE.AUTO
        elif mode == "Sample":return PCAMODE.SAMPLE
        elif mode=="Descriptor":return PCAMODE.DESCRIPTOR
        elif mode == "Cluster":return PCAMODE.CLUSTER
        elif mode == "Target":return PCAMODE.TARGET
    
    def is_vectorial_mode(self):
        return self in [PCAMODE.DESCRIPTOR,PCAMODE.CLUSTER, PCAMODE.TARGET]
class ComponentAnalysisParameters:
    def __init__(self,max_points:int =1000,number_of_correlation_to_show:int =0 ,mode = PCAMODE.AUTO):
        self.max_points:int = max_points
        self.number_of_correlation_to_show:int = number_of_correlation_to_show
        self.mode:PCAMODE  = PCAMODE.from_str(mode) if type(mode) is str else mode
@dataclass
class ComponentAnalysisOptions:
    pca_type:PCATYPE = PCATYPE.PCA
    descs:list=None
    ref:list=None
    trial:list=None
    axes:list=None
    ca_param:ComponentAnalysisParameters = None

