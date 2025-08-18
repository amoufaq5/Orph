import logging, sys
_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
def init_logger(name="orph", level=logging.INFO):
    lg = logging.getLogger(name); lg.setLevel(level)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter(_FMT)); lg.addHandler(h)
    return lg
