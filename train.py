import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import train_utils

args = None


def parse_args():
    """
    Parse and return command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train')

    # Basic Parameters
    parser.add_argument('--model_name', type=str, default='cnn_2d', help='The Name of the Model')
    parser.add_argument('--data_name', type=str, default='CWRUSlice', help='The Name of the Data')
    parser.add_argument('--data_dir', type=str, default="E:\Data\西储大学轴承数据中心网站",
                        help='The Directory of the Data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='Data Normalization Methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_A',
                        help='R_A: Random Split With Data Augmentation, R_NA: Random Split Without Data Augmentation, O_A: Order Split With Data Augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='Assign Device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='The Directory to Save the Model')
    parser.add_argument("--pretrained", type=bool, default=True, help='Whether to Load the Pretrained Model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size of the Training Process')
    parser.add_argument('--num_workers', type=int, default=0, help='The Number of Training Process')

    # Optimization Information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam', 'adamw', 'rmsprop'], default='adam',
                        help='The Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='The Initial Learning Rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='The Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='The Weight Decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'reduce', 'cosine', 'fix'], default='fix',
                        help='The Learning Rate Schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning Rate Scheduler Parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='The Learning Rate Decay (for step and stepLR)')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='Factor by Which the Learning Rate Will Be Reduced (For ReduceLROnPlateau)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of Epochs With No Improvement After Which Learning Rate Will Be Reduced (For ReduceLROnPlateau)')
    parser.add_argument('--T_max', type=int, default=10,
                        help='Maximum Number of Iterations (For Cosine Annealing Scheduler)')

    # Save, Load and Display Information
    parser.add_argument('--max_epoch', type=int, default=100, help='Max Number of Epoch')
    parser.add_argument('--print_step', type=int, default=100, help='The Interval of Log Training Information')
    args = parser.parse_args()
    return args


def set_cuda_device(cuda_device: str):
    """
    Set the CUDA device.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device.strip()


def prepare_save_dir(checkpoint_dir: str, model_name: str, data_name: str) -> str:
    """
    Prepare and return the saving path for the model.
    """
    sub_dir = f"{model_name}_{data_name}_{datetime.strftime(datetime.now(), '%m%d-%H%M%S')}"
    save_dir = os.path.join(checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def log_args(args, save_dir):
    """
    Log command line arguments.
    """
    # Construct the filename based on dataset name, optimizer, and learning rate
    filename = f"{args.data_name}_{args.opt}_{args.lr}.log"

    # Configure the logger with the new filename
    setlogger(os.path.join(save_dir, filename))
    # setlogger(os.path.join(save_dir, 'training.log'))
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")


if __name__ == '__main__':
    args = parse_args()

    set_cuda_device(args.cuda_device)

    save_dir = prepare_save_dir(args.checkpoint_dir, args.model_name, args.data_name)

    log_args(args, save_dir)

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()
