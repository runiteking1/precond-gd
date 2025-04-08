import os

def make_logger(preconditioner: str, threshold: float = 1e-1):
    if preconditioner in ["gn_exact", "kfac"]:
        log_name = f"{preconditioner}_thresh{threshold:.0e}.log"
    else:
        log_name = f"{preconditioner}.log"

    # Ensure all ranks append to the same file
    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        with open(log_name, "a") as f:
            f.write(msg + "\n")
            f.flush()

    return log, log_name.split('.')[0]

