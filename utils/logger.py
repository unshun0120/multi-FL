import os
from datetime import datetime

class Logger:
    def __init__(self, args, mode=""):
        self.log_dir = None
        self.log_file = None

        if hasattr(args, 'exp_timestamp') and args.exp_timestamp:
            timestamp = args.exp_timestamp
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if hasattr(args, 'model_path') and args.model_path:
            # for every new_client... .py file
            parent_dir = os.path.dirname(os.path.abspath(args.model_path))

            # parent_dir/new_client_{timestamp}
            master_dir = os.path.join(parent_dir, f"new_client_{timestamp}")
            
            # master_dir/{mode} 
            self.log_dir = os.path.join(master_dir, mode)   
        else: 
            # for FL training
            if hasattr(args, 'algo_name') and args.algorithm:
                algo_name = args.algorithm
            else:
                algo_name = mode if mode else "default"

            self.log_dir = os.path.join('logs', timestamp, algo_name)

        os.makedirs(self.log_dir, exist_ok=True)

        if hasattr(args, 'algorithm') and args.algorithm:
            self.log_file = os.path.join(self.log_dir, f'{args.algorithm}_result.log')
        else:
            self.log_file = os.path.join(self.log_dir, 'result.log')

        with open(self.log_file, 'w') as f:
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log Dir: {self.log_dir}\n")
            f.write("="*80)

    
    def log(self, message, print_to_console=True):
        if print_to_console:
            print(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")