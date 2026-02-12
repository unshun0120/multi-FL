import os
from datetime import datetime

class Logger:
    def __init__(self, args, mode=""):
        self.no_write_log = args.no_write_log
        self.log_dir = None
        self.log_file = None

        if not self.no_write_log: 
            if hasattr(args, 'exp_timestamp') and args.exp_timestamp:
                timestamp = args.exp_timestamp
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if hasattr(args, 'algo_name') and args.algo_name:
                algo_name = args.algo_name
            else:
                algo_name = mode if mode else "default"

            self.log_dir = os.path.join('logs', timestamp, algo_name)
            os.makedirs(self.log_dir, exist_ok=True)

            # log file
            self.log_file = os.path.join(self.log_dir, f'{algo_name}_result.log')

            # write config message
            with open(self.log_file, 'w') as f:
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log Dir: {self.log_dir}\n")
                f.write("="*100)

    def log(self, message, print_to_console=True):
        # 是否print在terminal
        if print_to_console:
            print(message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")

    def get_log_dir(self):
        return self.log_dir