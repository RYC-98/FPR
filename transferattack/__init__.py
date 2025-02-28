'''
simplified version

'''

from .collection.vitb import VITB
from .collection.vitb_gra import VITB_GRA
from .collection.vitb_sia import VITB_SIA
from .collection.vitb_vmi import VITB_VMI
from .collection.vitb_bsr import VITB_BSR
from .collection.deitb import DEITB
from .collection.pitt import PITT
from .collection.caits import CAITS



from .gradient.mifgsm import MIFGSM
from .gradient.vmifgsm import VMIFGSM
from .gradient.gra import GRA


from .input_transformation.sia import SIA
from .input_transformation.bsr import BSR


from .model_related.gns import GNS
from .model_related.pna_patchout import PNA_PatchOut
from .model_related.tgr import TGR

attack_zoo = {

            # collection
            'vitb':VITB,
            'vitb_gra':VITB_GRA,
            'vitb_sia':VITB_SIA,
            'vitb_vmi':VITB_VMI,
            'vitb_bsr':VITB_BSR,
            'deitb':DEITB,
            'pitt':PITT,
            'caits':CAITS,


            # gredient

            'mifgsm': MIFGSM,
            'vmifgsm': VMIFGSM,
            'gra': GRA,

            # input transformation
            'sia': SIA,
            'bsr': BSR,


            'gns': GNS,
            'pna_patchout': PNA_PatchOut,
            'tgr': TGR
        }

__version__ = '1.0.0'
