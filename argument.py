def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--max_steps', type=int, default=5000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_capacity', type=int, default=50000)
    parser.add_argument('--learning_starts', type=int, default=20000)
    parser.add_argument('--train_frequency', type=int, default=4)
    parser.add_argument('--target_update_frequency', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.0000625)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_steps', type=int, default=1000000)
    parser.add_argument('--model_path', type=str, default='dqn_model.pth')
    parser.add_argument('--best_model_path', type=str, default='dqn_model_best.pth')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=100,
                        help='Number of five-life evaluation episodes')
    parser.add_argument('--escape_stagnation_steps', type=int, default=1200,
                        help='Evaluation steps without positive reward before a paddle sweep')
    parser.add_argument('--escape_cooldown_steps', type=int, default=600,
                        help='Minimum steps between evaluation escape maneuvers')
    return parser
