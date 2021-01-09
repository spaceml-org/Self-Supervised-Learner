from ssl_dali_distrib import SimCLR
from transforms_dali import SimCLRTrainDataTransform
from encoders_dali import load_encoder


def cli_main(size, DATA_PATH, batch_size, num_workers, hidden_dims, epochs, lr, 
          patience, val_split, withhold, gpus, encoder, log_name, online_eval):
    
    wandb_logger = WandbLogger(name=log_name,project='SpaceForce')
    checkpointed = '.ckpt' in encoder    
    if checkpointed:
        print('Resuming SSL Training from Model Checkpoint')
        try:
            model = SIMCLR.load_from_checkpoint(checkpoint_path=encoder)
            embedding_size = model.embedding_size
        except Exception as e:
            print(e)
            print('invalid checkpoint to initialize SIMCLR. This checkpoint needs to include the encoder and projection and is of the SIMCLR class from this library. Will try to initialize just the encoder')
            checkpointed = False 
            
    elif not checkpointed:
        encoder, embedding_size = load_encoder(encoder)
        model = SIMCLR(size = size, encoder = encoder, embedding_size = embedding_size, gpus = gpus, epochs = epochs, DATA_PATH = DATA_PATH, withhold = withhold, batch_size = batch_size, val_split = val_split, hidden_dims = hidden_dims, train_transform = SimCLRTrainDataTransform, val_transform = SimCLRTrainDataTransform, num_workers = num_workers, lr = lr)
        
    online_evaluator = SSLOnlineEvaluator(
      drop_p=0.,
      hidden_dim=None,
      z_dim=embedding_size,
      num_classes=model.num_classes,
      dataset='None'
    )
    
    cbs = []
    backend = 'dp'
    
    if patience > 0:
        cb = EarlyStopping('val_loss', patience = patience)
        cbs.append(cb)
    
    if online_eval:
        cbs.append(online_evaluator)
        backend = 'ddp'
        
    trainer = Trainer(gpus=gpus, max_epochs = epochs, progress_bar_refresh_rate=5, callbacks = cbs, distributed_backend=f'{backend}' if gpus > 1 else None, logger = wandb_logger, enable_pl_optimizer=True)
    
    print('USING BACKEND______________________________ ', backend)
    trainer.fit(model)
    Path(f"./models/SSL").mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"./models/SSL/{log_name}")
    #torch.save(model.encoder.state_dict(), f"./models/SSL/SIMCLR_SSL_{version}/SIMCLR_SSL_{version}.pt")
    return f"./models/SSL/{log_name}"
if __name__ == '__main__':
    cli_main()