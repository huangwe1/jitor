class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = r'C:\Users\huangjian\Desktop\All-in-One-main'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'

        self.lasot_dir = r'D:\datasets\LaSOT'
        self.got10k_dir = r'D:\datasets\GOT-10k\train'
        self.got10k_val_dir = r'D:\datasets\GOT-10k\val'
        self.trackingnet_dir = r'D:\datasets\TrackingNet'
        self.coco_dir = r'D:\datasets\COCO'
        self.refcoco_dir = r'D:\datasets\COCO'
        self.visualgenome_dir = r'D:\datasets\VisualGenome'
        self.imagenet_dir = r'D:\datasets\ILSVRC2015'

        self.tnl2k_dir = r'D:\datasets\TNL2K\train'
        self.otb99lang_dir = r'D:\datasets\OTB99-LANG\train'
        self.webuav3m_dir = r'D:\datasets\WebUAV-3M\Train'

        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_lmdb_dir = ''
        self.imagenet_lmdb_dir = ''

        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
