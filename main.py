from config import CFG

if __name__ == "__main__":
    config_file = "/Users/ronghao/code/stitch/pystitch/configfile"
    cfg = CFG()
    cfg.from_config_yaml(config_path=config_file)
