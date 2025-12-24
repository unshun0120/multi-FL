import argparse

def get_config():
    parser = argparse.ArgumentParser()
 
    # --- Basic setup ---
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_write_log', action='store_true', help='logging and plotting to files')
    parser.add_argument('--no_plot_log', action='store_true', help='logging and plotting to files')
    parser.add_argument('--no_save_model', action='store_true', help='Whether to save the trained model checkpoints')
    parser.add_argument('--save_model_epoch', type=int, default=5, help='The frequency to save the trained model checkpoints')


    # --- Dataset ---
    parser.add_argument("--num_mnist", type=int, default=11)
    parser.add_argument("--num_emnist", type=int, default=11)
    parser.add_argument("--num_fashionmnist", type=int, default=11)
    parser.add_argument("--num_cifar10", type=int, default=11)
    parser.add_argument("--num_cifar100", type=int, default=11)
    parser.add_argument('--num_cifar100_super', type=int, default=11)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)


    # --- Model Training setting ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument("--global_feature_dim", type=int, default=256, help='= generator feature dimension = classifier input dimension = adapter layer output dimension')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer')

    # --- Client ---
    parser.add_argument('--num_new_clients', type=int, default=1, help='Number of IID clients for generalization test')
    parser.add_argument("--num_local_noise", type=int, default=10)
    parser.add_argument("--local_relation_weight", type=float, default=0.5, help='Weight for relation distillation loss')
    parser.add_argument('--client_lr', type=float, default=0.001, help='Learning rate for client local training')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs per round')


    # --- Server ---
    parser.add_argument('--global_rounds', type=int, default=100, help='Total communication rounds')
    parser.add_argument('--gen_optim', type=str, default='Adam', help='Generator Optimizer')
    parser.add_argument('--gen_lr', type=float, default=0.001, help='Learning rate of Generator')
    parser.add_argument('--server_gen_epochs', type=int, default=20, help='Generator training epochs per round')
    parser.add_argument("--gen_observer_weight", type=float, default=0.5, help='Weight for generator observer loss')



    args = parser.parse_args()

    return args