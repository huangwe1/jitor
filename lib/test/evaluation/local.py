from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # ============ 请修改以下路径为你的实际路径 ============

    prj_dir = r'C:\Users\huangjian\Desktop\All-in-One-main'

    settings.prj_dir = prj_dir
    settings.network_path = prj_dir + '/output'
    settings.results_path = prj_dir + '/output/test/tracking_results'
    settings.result_plot_path = prj_dir + '/output/test/result_plots'
    settings.segmentation_path = prj_dir + '/output/test/segmentation_results'
    settings.save_dir = prj_dir + '/output'

    # ============ 测试数据集路径 ============
    settings.lasot_path = r'D:\datasets\LaSOT'
    settings.lasotext_path = r'D:\datasets\LaSOT_extension'
    settings.got10k_path = r'D:\datasets\GOT-10k\test'
    settings.trackingnet_path = r'D:\datasets\TrackingNet'
    settings.tnl2k_path = r'D:\datasets\TNL2K\test'
    settings.otb99lang_path = r'D:\datasets\OTB99-LANG\test'
    settings.webuav3m_path = r'D:\datasets\WebUAV-3M\Test'

    # 以下一般不需要修改
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.lasot_lmdb_path = ''

    return settings
