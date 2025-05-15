class _DefaultConfig:
    """
    Model config.
    """

    def __init__(self):
        self.n_args = 7
        self.only_black_white_color = False
        self.use_cont_path_idx = True

        self.add_style_token = False
        self.signature = "IconShop_diffvg_select_conti"

        self.similarity_threshold = 0.1
        self.select_threshold = 0

        self.pre_nlayer = 18
        self.feature_dim = 2048
        self.dim_transformer = 800
        self.nhead = 12
        self.nlayer = 28

        self.learning_rate = 0.00003
        self.loss_type = "x0rec"

        self.conditioning_dropout_prob = 0.1

        self.use_glb = False
        self.use_LCScheduler = True
        self.discrete_t = False
        self.use_norm_out = False

        self.pretrained_fn = "2400"

        self.ddim_num_steps = 100
        # ddpm, ddim
        self.samp_method = "ddpm"
        self.guidance_scale = 5.0

        self.batch_size = 4

        self.loader_num_workers = 2

        self.max_total_len = 401
        self.max_points = 401
        self.max_paths_len_thresh = 31

        self.dittype = "diffusers"

        self.xy_scale = 1.0

        self.label_condition = True

        self.use_sum_loss = True
