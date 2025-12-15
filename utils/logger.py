import os
from datetime import datetime

class Logger:
    def __init__(self, args):
        self.write_log = args.write_log
        self.log_dir = None
        self.log_file = None

        if self.write_log:
            # log folder 名字用當下時間命名
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join('logs', timestamp)
            if not os.path.exists(self.log_dir): 
                os.makedirs(self.log_dir)

            # log file
            self.log_file = os.path.join(self.log_dir, 'FL_train_result.log')

            # write start message
            with open(self.log_file, 'w') as f:
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Log Dir: {self.log_dir}\n")
                f.write(f"Config Settings:\n")
                for arg, value in vars(args).items():
                    f.write(f"  {arg}: {value}\n")
                f.write("="*100 + "\n\n")

    def log(self, message, print_to_console=True):
        # print 在 terminal
        if print_to_console:
            print(message)
        
        if self.write_log and self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{message}\n")

    def get_log_dir(self):
        return self.log_dir