WEB_DIRECTORY = "./web"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", ]

import sys; print(sys.executable)

from .NodeBasic.C_image import Image_solo_crop, Image_solo_stitch, Mask_simple_adjust

#-load------------------------------------------#



NODE_CLASS_MAPPINGS= {
"Mask_simple_adjust":Mask_simple_adjust,
"Image_solo_crop": Image_solo_crop,  
"Image_solo_stitch": Image_solo_stitch,    
}



NODE_DISPLAY_NAME_MAPPINGS = {
"DoWhileEnd": "Do While End",
"MultiParamFormula": "Multi Param Formula",
"flow_for_start": "Flow For Start",
"flow_for_end": "Flow For End",
"Data_Highway": "数据_高速通道"



}











