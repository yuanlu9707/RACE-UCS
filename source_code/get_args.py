import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 已经验证这里的参数可被存入params.json

    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--test', action='store_true', default=False, )
    parser.add_argument('--test-with-shenbi', action='store_true')
    parser.add_argument('--test-save-heatmap', action='store_true')
    parser.add_argument('--user', type=str, default='yyx')
    parser.add_argument('--env', type=str, default='Mobile')
    parser.add_argument('--algo', type=str, choices=['IPPO'], default='IPPO',
                        help='IPPO/MAPPO backbone with optional RITRL and CRE')
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument("--dataset", type=str, default='purdue', choices=['NCSU', 'purdue'])
    parser.add_argument("--poi_num", type=int, default=48)
    parser.add_argument("--tag", type=str, default='', help='每个单独实验的备注')
    # dirs
    parser.add_argument("--output_dir", type=str, default='runs/debug', help="which fold to save under 'runs/'")
    parser.add_argument('--group', type=str, default='debug', help='填写我对一组实验的备注，作用与wandb的group和tb的实验保存路径')
    # system stub
    parser.add_argument('--mute_wandb', default=True, action='store_false')  # 后期服务器联网不稳定 网断了进程就会sleep 不要用wandb！
    # tune agent
    parser.add_argument('--checkpoint', type=str)  # load pretrained model
    parser.add_argument('--n_thread', type=int, default=16)
    # parser.add_argument('--n_thread', type=int, default=16)
    parser.add_argument('--n_iter', type=int, default=5000)
    # tune algo
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_v', type=float)
    parser.add_argument('--lr_colla', type=float)
    parser.add_argument('--use-stack-frame', action='store_true')
    # parser.add_argument('--use_extended_value', action='store_false', help='反逻辑，仅用于DPPO')
    # parser.add_argument('--use-mlp-model', action='store_true', help='将model改为最简单的mlp，仅用于DMPO')
    # parser.add_argument('--multi-mlp', action='store_true', help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    parser.add_argument('--g2a_hidden_dim', type=int, default=64, help='在model中分开预测obs中不同类别的信息，仅用于DMPO')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--map_size', type=int, default=6)  # hyper
    parser.add_argument('--g2a_hops', type=int, default=1)  # hyper
    parser.add_argument('--update_colla_by_v_0307', action='store_true')  # hyper

    # tune env
    ## setting
    parser.add_argument('--fixed-range', action='store_false')  # 重要，sensing range现在固定了
    parser.add_argument('--collect_range', type=float, default=500)
    parser.add_argument('--dyna_level', type=str, default='2', help='指明读取不同难度的poi_QoS.npy')
    parser.add_argument('--init_energy', type=float, default=1000000) #719280
    parser.add_argument('--w_noise', type=float, default=-107)  # 0222morning determined 用不到
    parser.add_argument('--user_data_amount', type=float, default=3) #单位-1Mbits
    parser.add_argument('--update_num', type=int, default=10)  # 两个数据集统一将default设为10

    parser.add_argument('--uav_num', type=int, default=5)
    parser.add_argument('--fixed-col-time', action='store_false')
    parser.add_argument('--aoith', default=30, type=int)  # 0222morning determined
    parser.add_argument('--txth', default=1, type=int) #传输要求，没啥意思，可改
    parser.add_argument('--r_th_vio', default=True) #r-th_vio 吞吐量满意率的奖励惩罚
    parser.add_argument('--reward_add_r_th_vio', default=False)
    parser.add_argument('--uav_height', default=100, type=int)
    parser.add_argument('--hao02191630', action='store_false')
    parser.add_argument('--always_fixed_antenna02230040', default=-1, type=int)

    ## MDP
    parser.add_argument('--max_episode_step', type=int, default=120)
    parser.add_argument('--MAPPO', default=False)
    parser.add_argument('--obs_add_action', default=False)
    parser.add_argument('--ues_GRU', default=False,
                        help='use a local IPPO actor: observation -> MLP -> GRUCell -> action heads')
    parser.add_argument('--future_obs', type=int, default=0)
    # RITRL uses IPPO actors and optionally the centralized MAPPO critic.
    parser.add_argument('--use_ritrl', default=True,
                        help='enable recurrent reconstruction, contrastive and team consensus learning')
    parser.add_argument('--ritrl_hidden_dim', type=int, default=64)
    parser.add_argument('--ritrl_id_dim', type=int, default=16)
    parser.add_argument('--ritrl_action_embed_dim', type=int, default=16)
    parser.add_argument('--ritrl_projection_dim', type=int, default=32)
    parser.add_argument('--ritrl_consensus_classes', type=int, default=8)
    parser.add_argument('--ritrl_consensus_embed_dim', type=int, default=16)
    parser.add_argument('--ritrl_lr', type=float, default=3e-4)
    parser.add_argument('--ritrl_rec_weight', type=float, default=1.0)
    parser.add_argument('--ritrl_lambda0', type=float, default=0.1,
                        help='reconstruction weight for unobserved user/UAV slots; must be in (0, 1)')
    parser.add_argument('--ritrl_cl_weight', type=float, default=0.1,
                        help='weight of individual InfoNCE: g_rec(d) versus g_tar(stop_gradient(h))')
    parser.add_argument('--ritrl_cons_weight', type=float, default=0.1)
    parser.add_argument('--ritrl_cl_temperature', type=float, default=0.1)
    parser.add_argument('--ritrl_online_temperature', type=float, default=0.1)
    parser.add_argument('--ritrl_target_temperature', type=float, default=0.04)
    parser.add_argument('--ritrl_ema', type=float, default=0.99)
    parser.add_argument('--ritrl_center_momentum', type=float, default=0.9)
    parser.add_argument('--ritrl_aux_epochs', type=int, default=1)
    parser.add_argument('--ritrl_sequence_length', type=int, default=16,
                        help='auxiliary truncated backpropagation sequence length')
    parser.add_argument('--ritrl_grad_clip', type=float, default=10.0)
    parser.add_argument('--ritrl_rec_warmup', type=int, default=0,
                        help='rollout updates for reconstruction-only pretraining (PPO paused)')
    parser.add_argument('--ritrl_cl_warmup', type=int, default=0,
                        help='additional rollout updates for reconstruction + contrastive pretraining')
    parser.add_argument('--use_snrmap', action='store_true', default=False)  # shrotcut is always used
    # CRE adds an intrinsic reward; RITRL's own losses stay unchanged.
    parser.add_argument('--use_cre', default=True,
                        help='enable CRE (requires RITRL); accepts True/False')
    parser.add_argument('--cre_hidden_dim', type=int, default=64)
    parser.add_argument('--cre_state_dim', type=int, default=32)
    parser.add_argument('--cre_latent_dim', type=int, default=16)
    parser.add_argument('--cre_context_dim', type=int, default=32)
    parser.add_argument('--cre_action_embed_dim', type=int, default=16)
    parser.add_argument('--cre_lr', type=float, default=3e-4)
    parser.add_argument('--cre_score_lr', type=float, default=3e-4,
                        help='learning rate for the separate -E[log gamma] scorer update')
    parser.add_argument('--cre_norm_discount', type=float, default=0.99,
                        help='discount used only by intrinsic return normalization')
    parser.add_argument('--cre_norm_min_std', type=float, default=1e-2,
                        help='minimum shared running standard deviation for intrinsic reward scaling')
    parser.add_argument('--cre_reward_weight', type=float, default=0.01)
    parser.add_argument('--cre_alpha_I', type=float, default=1.0,
                        help='alpha_I in L_CRE = L_pred + alpha_I * (L_UB + L_LB)')
    parser.add_argument('--cre_rho', type=float, default=0.1)
    parser.add_argument('--cre_tail_probability', type=float, default=0.1)
    parser.add_argument('--cre_cmin', type=float, default=0.0)
    parser.add_argument('--cre_cmax', type=float, default=10.0)
    parser.add_argument('--cre_epsilon', type=float, default=1e-6)
    parser.add_argument('--cre_temperature', type=float, default=0.1)
    parser.add_argument('--cre_negatives', type=int, default=32)
    parser.add_argument('--cre_memory_size', type=int, default=2048)
    parser.add_argument('--cre_batch_size', type=int, default=256)
    parser.add_argument('--cre_epochs', type=int, default=1)
    parser.add_argument('--cre_ema', type=float, default=0.99)
    parser.add_argument('--cre_grad_clip', type=float, default=10.0)
    parser.add_argument('--high_level_dont_use_snrmap', action='store_true')  #
    parser.add_argument('--high_level_knn_coefficient', type=float, default=-1)  #
    parser.add_argument('--aVPS', type=float, default=0.2)
    parser.add_argument('--tVPS', type=float, default=0.2)
    parser.add_argument('--agent_field', type=float, default=500)
    input_args = parser.parse_args()

    # Keep the default=True argument style while accepting explicit CLI booleans.
    if isinstance(input_args.use_ritrl, str):
        value = input_args.use_ritrl.lower()
        if value not in ('true', 'false'):
            parser.error('--use_ritrl must be True or False')
        input_args.use_ritrl = value == 'true'

    if isinstance(input_args.use_cre, str):
        if input_args.use_cre.lower() not in ('true', 'false'):
            parser.error('--use_cre must be True or False')
        input_args.use_cre = input_args.use_cre.lower() == 'true'
    if input_args.use_cre:
        if not input_args.use_ritrl:
            parser.error('--use_cre requires --use_ritrl True')
        for name in ('hidden_dim', 'state_dim', 'latent_dim', 'context_dim', 'action_embed_dim',
                     'negatives', 'memory_size', 'batch_size', 'epochs'):
            if getattr(input_args, 'cre_' + name) < 1:
                parser.error('--cre_' + name + ' must be positive')
        for name in ('lr', 'score_lr', 'norm_min_std', 'epsilon', 'temperature', 'grad_clip'):
            if not 0 < getattr(input_args, 'cre_' + name) < float('inf'):
                parser.error('--cre_' + name + ' must be finite and positive')
        for name in ('reward_weight', 'rho', 'alpha_I'):
            if not 0 <= getattr(input_args, 'cre_' + name) < float('inf'):
                parser.error('--cre_' + name + ' must be finite and nonnegative')
        if not 0 < input_args.cre_tail_probability < 1:
            parser.error('--cre_tail_probability must be in (0, 1)')
        if not 0 <= input_args.cre_ema < 1:
            parser.error('--cre_ema must be in [0, 1)')
        if not 0 <= input_args.cre_norm_discount < 1:
            parser.error('--cre_norm_discount must be in [0, 1)')
        if not -float('inf') < input_args.cre_cmin < input_args.cre_cmax < float('inf'):
            parser.error('CRE requires finite cre_cmin < cre_cmax')

    if input_args.use_ritrl:
        if not 0 < input_args.ritrl_lambda0 < 1:
            parser.error('--ritrl_lambda0 must be in (0, 1)')
        if input_args.algo != 'IPPO':
            parser.error('--use_ritrl requires --algo IPPO; use --MAPPO True for a centralized critic')
        if input_args.uav_num < 2:
            parser.error('--use_ritrl requires at least two UAVs')
        input_args.ues_GRU = True  # activate recurrent rollout storage/reset hooks
        for name in ('hidden_dim', 'id_dim', 'action_embed_dim', 'projection_dim',
                     'consensus_embed_dim', 'aux_epochs', 'sequence_length'):
            if getattr(input_args, 'ritrl_' + name) < 1:
                parser.error('--ritrl_' + name + ' must be positive')
        if input_args.ritrl_consensus_classes < 2:
            parser.error('--ritrl_consensus_classes must be at least 2')
        for name in ('lr', 'cl_temperature', 'online_temperature', 'target_temperature', 'grad_clip'):
            if not 0 < getattr(input_args, 'ritrl_' + name) < float('inf'):
                parser.error('--ritrl_' + name + ' must be finite and positive')
        for name in ('ema', 'center_momentum'):
            if not 0 <= getattr(input_args, 'ritrl_' + name) < 1:
                parser.error('--ritrl_' + name + ' must be in [0, 1)')
        for name in ('rec_weight', 'cl_weight', 'cons_weight', 'rec_warmup', 'cl_warmup'):
            if not 0 <= getattr(input_args, 'ritrl_' + name) < float('inf'):
                parser.error('--ritrl_' + name + ' must be finite and nonnegative')
        if input_args.ritrl_rec_weight + input_args.ritrl_cl_weight + input_args.ritrl_cons_weight == 0:
            parser.error('at least one RITRL loss weight must be positive')

    if input_args.ues_GRU and input_args.algo != 'IPPO':
        parser.error('--ues_GRU currently supports --algo IPPO only')


    if input_args.test_with_shenbi:
        input_args.test = True


    if input_args.high_level_dont_use_snrmap:
        input_args.use_snrmap = False
    if input_args.high_level_knn_coefficient != -1:
        input_args.knn_coefficient = input_args.high_level_knn_coefficient

    if input_args.debug:
        input_args.group = 'debug'
        input_args.n_thread = 2
    input_args.output_dir = f'runs/{input_args.group}'

    if input_args.test:
        input_args.group = 'test'
        input_args.n_thread = 1
        input_args.output_dir = f'{input_args.checkpoint}/test'


    if input_args.dataset == 'NCSU':  # 在NCSU的默认值
        if input_args.poi_num == 116:
            input_args.poi_num = 48

    if input_args.dataset == 'purdue':  # 在NCSU的默认值
        if input_args.poi_num == 116:
            input_args.poi_num = 59

    env_args = {
        "max_episode_step": input_args.max_episode_step,
        "collect_range": input_args.collect_range,
        "initial_energy": input_args.init_energy,
        "user_data_amount": input_args.user_data_amount,
        "update_num": input_args.update_num,
        "uav_num": input_args.uav_num,
        "AoI_THRESHOLD": input_args.aoith,
        "RATE_THRESHOLD": input_args.txth,
        "uav_height": input_args.uav_height,
        "aoi_vio_penalty_scale": input_args.aVPS,
        "hao02191630": input_args.hao02191630,
        "w_noise": input_args.w_noise,
        "agent_field": input_args.agent_field,
    }
    if input_args.poi_num is not None:
        env_args["poi_num"] = input_args.poi_num

    return input_args, env_args
