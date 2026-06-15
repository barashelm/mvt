import torch
import torch.nn as nn
import time

from utils.parser import args
from utils import logger, Trainer, Tester, L1ALoss
from utils import init_device, init_model, WarmUpCosineAnnealingLR, FakeLR
from dataset import Cost2100DataLoader


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()

    # Define model
    # models available:'csinetp', 'crnet', 'transnet', 'stnet', 'mnet', 'crissnet'
    modename = 'transnet'
    model = init_model(modename, args)

    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)
    # criterion = L1ALoss(lamda_start=0e-2).to(device)


    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        return

    # Define optimizer and scheduler
    if modename == 'csinetp':
        lr_init = 1e-3
    elif modename == 'crnet':
        lr_init = 2e-3
    elif modename == 'acrnet':
        lr_init = 4e-3
    elif modename == 'transnet':
        lr_init = 1e-4
    elif modename == 'stnet':
        lr_init = 1e-3
    elif modename == 'mnet':
        lr_init = 2e-3
    elif modename == 'crissnet':
        lr_init = 3e-3
    else: lr_init = 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    if modename in ('crnet', 'crissnet', 'mvt'):
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)
    else:
        scheduler = FakeLR(optimizer=optimizer)

    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume)

    # Start training
    start_time = time.time()
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)
    end_time = time.time()
    training_time = end_time - start_time

    # Final testing
    loss, nmse, qcs = Tester(model, device, criterion)(test_loader)
    print(f"\n=! Final test loss: {loss:.3e}"
          f"\n         test NMSE: {nmse:.3e}"
          f"\n         test QCS: {qcs:.3e}"
        #   f"\n         test SGCS: {qcs ** 2:.3e}\n"
          f"\n training time: {training_time:.3e} sec\n")

    # torch.save(model.state_dict(), f'{modename}_{args.scenario}_cr{args.cr}.pth')

if __name__ == "__main__":
    main()
