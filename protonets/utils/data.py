import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'xbd':
        ds = protonets.data.xbd.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
