import argparse
import os

if __name__ == "__main__":
    """fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)"""

    parser = argparse.ArgumentParser(description="ClinStab robustness training and evaluation")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Crossformer, FEDformer, Informer, iTransformer, MTST, Nonstationary_Transformer, PatchTST, Reformer, Transformer, TCN, Medformer, ClinStab]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )

    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (percent)"
    )

    parser.add_argument('--stddev', type=float, default=2, help='噪声扰动幅度')
    parser.add_argument('--eps', type=float, default=0.1, help='PGD攻击幅度')
    parser.add_argument('--alpha', type=float, default=0.01, help='PGD攻击步长')
    parser.add_argument('--steps', type=int, default=20, help='PGD攻击次数')
    parser.add_argument('--w_adv', type=float, default=1, help='PGD前后输出接近')
    parser.add_argument('--w_cls', type=float, default=0.5, help='teacher的输出和原本的输出距离接近')
    parser.add_argument('--w_sim', type=float, default=0.5, help='IB输出特征间的loss')
    parser.add_argument('--w_ib', type=float, default=1.0, help='IB的权重')
    parser.add_argument('--iters_per_epoch', type=int, default=10,
                    help='每个epoch内对同一扰动数据训练的迭代次数')
    # model define for baselines
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16, help="chunk_size used in LightTS"
    )
    parser.add_argument(
        "--patch_len", type=int, default=16, help="patch_len used in PatchTST"
    )
    parser.add_argument("--stride", type=int, default=8, help="stride used in PatchTST")
    parser.add_argument(
        "--sampling_rate", type=int, default=256, help="frequency sampling rate"
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="2,4,8",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip,frequency,jitter,mask,channel,drop",
        help="A comma-seperated list of augmentation types (none, jitter or scale). "
             "Randomly applied to each granularity. "
             "Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )
    parser.add_argument(
        "--scdm_alpha",
        type=float,
        default=4.67,
        help="SCDM sparsity ratio; k=floor(channels/scdm_alpha)",
    )

    # optimization
    # parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument(
        "--num_workers", type=int, default=0, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="use stochastic weight averaging",
        default=False,
    )


    parser.add_argument('--use_csam', action='store_true', help='使用CSAM注意力机制')
    parser.add_argument('--use_csam_slice', action='store_true', help='在CSAM中使用切片注意力')
    parser.add_argument('--use_csam_uncertainty', action='store_true', help='在CSAM中使用不确定性')
    parser.add_argument('--csam_rank', type=int, default=5, help='CSAM中不确定性模型的秩')
    parser.add_argument('--use_shvit', action='store_true', help='使用SHVIT注意力机制')
    parser.add_argument('--use_pola_attn', action='store_true',
                        help='使用PolarFormer注意力增强',
                        default=False)
    parser.add_argument('--pola_kernel', type=int, default=5,
                        help='PolarFormer的卷积核大小')
    parser.add_argument('--pola_alpha', type=int, default=4,
                        help='PolarFormer的alpha参数')

    # parser.add_argument('--des', type=str, default='test', help='exp description')
    # parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--rho', type=float, default=0.75,
                    help='AutoPGD步长调整参数，控制步长增减比例')
    parser.add_argument('--max_restarts', type=int, default=2,
                    help='AutoPGD最大重启次数，当攻击停滞时重新初始化')

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus"
    )
    # parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

    # de-stationary projector params
    parser.add_argument(
        "--p_hidden_dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="hidden layer dimensions of projector (List)",
    )
    parser.add_argument(
        "--p_hidden_layers",
        type=int,
        default=2,
        help="number of hidden layers in projector",
    )


    args = parser.parse_args()

    # Delay optional scientific imports until after argparse handles --help.
    import torch
    import random
    import numpy as np
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from exp.exp_imputation import Exp_Imputation
    from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
    from exp.exp_anomaly_detection import Exp_Anomaly_Detection
    from exp.exp_classification import Exp_Classification
    from exp.exp_classification_rob import Exp_Classification_Rob
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp_Rob = None
    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
        Exp_Rob = Exp_Classification_Rob
    else:
        Exp = Exp_Long_Term_Forecast
    if args.is_training:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


            # setting record of experiments
            args.seed = seed
            setting = "{}/{}/{}/{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.seed,
            )
            pretrained_model_path = os.path.join('./checkpoints/', setting, 'checkpoint.pth')
            if os.path.exists(pretrained_model_path):
                print(f"Using pretrained model from: {pretrained_model_path}")
                exp = Exp(args)  # 设置原始实验
                exp.test(setting, test=1)
            else:
                print(f'pretrained_model_path not found: {pretrained_model_path}')
                exp = Exp(args)  # 设置原始实验
                print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
                exp.train(setting)

                print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                exp.test(setting)

            # 添加鲁棒性训练部分
            if Exp_Rob:  # 如果存在对应的鲁棒训练类
                print(">>>>>>>start robust training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))

                if os.path.exists(pretrained_model_path):
                    print(f"Using pretrained model from: {pretrained_model_path}")
                    experiment_rob = Exp_Rob(args)
                    experiment_rob.train(setting, pretrained_model_path)

                    # 测试鲁棒性模型
                    robust_model_path = os.path.join('./checkpoints/', setting, 'rob_checkpoint.pth')
                    print(">>>>>>>testing robust model : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
                    if os.path.exists(robust_model_path):
                        experiment_rob.test(setting, model_path=robust_model_path, test=1)
                    else:
                        print(f"Warning: Robust model not found at {robust_model_path}")
                else:
                    print(f"Warning: Pretrained model not found at {pretrained_model_path}")

            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            seed = 41 + ii
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # comment out the following lines if you are using dilated convolutions, e.g., TCN
            # otherwise it will slow down the training extremely
            if args.model != "TCN":
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

            args.seed = seed
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_seed{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.seed,
            )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
