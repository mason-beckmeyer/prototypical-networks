import os
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt
from protonets.engine import Engine
from protonets.utils.model import load as load_model
from protonets.data.xbd import load as load_xbd


def main(opt):
    os.makedirs(opt['log.exp_dir'], exist_ok=True)

    # Save config
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f, indent=2)

    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['use_cuda']:
        torch.cuda.manual_seed(1234)

    data = load_xbd(opt, ['train', 'test'])
    train_loader = data['train']
    val_loader = data['test']

    model = load_model(opt)
    if opt['use_cuda']:
        model.cuda()

    engine = Engine()
    meters = {
        'train': {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']},
        'test': {field: tnt.meter.AverageValueMeter() for field in opt['log.fields']}
    }

    def on_start(state):
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split in meters:
            for meter in meters[split].values():
                meter.reset()
        state['scheduler'].step()
    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field in meters['train']:
            meters['train'][field].add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(_, state):
        from protonets.utils.model import evaluate
        evaluate(state['model'], val_loader, meters['test'], desc=f"Epoch {state['epoch']} validation")

        print(f"Epoch {state['epoch']:02d}:", {k: f"{v.mean:.4f}" for k, v in meters['val'].items()})

        model_path = os.path.join(opt['log.exp_dir'], 'best_model.pt')
        torch.save(state['model'].cpu(), model_path)
        if opt['use_cuda']:
            state['model'].cuda()
    engine.hooks['on_end_epoch'] = lambda state: on_end_epoch({}, state)

    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, opt['optim_method']),
        optim_config={'lr': opt['lr'], 'weight_decay': opt['weight_decay']},
        max_epoch=opt['epochs']
    )
