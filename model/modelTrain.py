# =======================================================================================================================
# =======================================================================================================================
import os
import numpy as np
import h5py
import logging

from datetime import datetime
from modelDesign import *
from torch.utils.tensorboard import SummaryWriter



# Parameters Setting
# ========================================================
def load_param(case):
    if case == '1a' or case == '1b':
        ## case1
        return dict(
            NUM_SUBCARRIERS = 624,
            NUM_OFDM_SYMBOLS = 12,
            NUM_LAYERS = 2,
            NUM_BITS_PER_SYMBOL = 4,
        )
    else:
        ## case2
        return dict(
            NUM_SUBCARRIERS = 96,
            NUM_OFDM_SYMBOLS = 12,
            NUM_LAYERS = 4,
            NUM_BITS_PER_SYMBOL = 6,
        )

# Data Loading
# ========================================================
def load_data(train_dataset_dir, case):
    print(f'=====================load case{case} data===============')
    f = h5py.File(os.path.join(train_dataset_dir, f"D{case[0]}.hdf5"), 'r')
    rx_signal = f['rx_signal'][:]
    tx_bits = f['tx_bits'][:]
    pilot = f['pilot'][:]
    f.close()
    print('rx_signal:', rx_signal.shape, rx_signal.dtype)
    print('tx_bits:', tx_bits.shape, tx_bits.dtype)
    samples = rx_signal.shape[0]
    pilot = np.tile(pilot, [samples, 1, 1, 1, 1])
    print('pilot:', pilot.shape, pilot.dtype)

    rx_signal_train = rx_signal[:int(rx_signal.shape[0] * 0.99)]
    rx_signal_val = rx_signal[int(rx_signal.shape[0] * 0.99):]
    pilot_train = pilot[:int(pilot.shape[0] * 0.99)]
    pilot_val = pilot[int(pilot.shape[0] * 0.99):]
    tx_bits_train = tx_bits[:int(tx_bits.shape[0] * 0.99)]
    tx_bits_val = tx_bits[int(tx_bits.shape[0] * 0.99):]

    print('rx_signal_train:', rx_signal_train.shape, rx_signal_train.dtype)
    print('tx_bits_train:', tx_bits_train.shape, tx_bits_train.dtype)
    print('pilot_train:', pilot_train.shape, pilot_train.dtype)

    print('rx_signal_val:', rx_signal_val.shape, rx_signal_val.dtype)
    print('tx_bits_val:', tx_bits_val.shape, tx_bits_val.dtype)
    print('pilot_val:', pilot_val.shape, pilot_val.dtype)

    return rx_signal_train, tx_bits_train, pilot_train, rx_signal_val, tx_bits_val, pilot_val


def generator(batch, rx_signal_in, pilot_in, tx_bits_in):
    idx_tmp = np.random.choice(rx_signal_in.shape[0], batch, replace=False)
    batch_rx_signal = rx_signal_in[idx_tmp].astype(np.float32)
    batch_pilot = pilot_in[idx_tmp].astype(np.float32)
    batch_tx_bits = tx_bits_in[idx_tmp].astype(np.float32)
    return torch.from_numpy(batch_rx_signal), torch.from_numpy(batch_pilot), torch.from_numpy(batch_tx_bits)

# Model Constructing
# ========================================================
def build_model(case):
    p = load_param(case)
    Model = Neural_receiver(
        subcarriers=p['NUM_SUBCARRIERS'],
        timesymbols=p['NUM_OFDM_SYMBOLS'], streams=p['NUM_LAYERS'],
        num_bits_per_symbol=p['NUM_BITS_PER_SYMBOL'])
        
    return Model

# Model Training and Saving
# =========================================================
def train(train_dataset_dir, case, EPOCHS):
    rx_signal_train, tx_bits_train, pilot_train, rx_signal_val, tx_bits_val, pilot_val = load_data(train_dataset_dir, case)
    Model = build_model(case)
    criterion = nn.BCEWithLogitsLoss()
    
    log_dir = f'./logs/case{case}/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    model_save_dir = './logs/models'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    writer = SummaryWriter(log_dir=log_dir)

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(f'{log_dir}/log.txt')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    if torch.cuda.is_available():
        Model = Model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(Model.parameters(), lr=1e-2)

    bestLoss = 100
    for epoch in range(EPOCHS):
        Model.train()
        ModelInput1, ModelInput2, label = generator(16, rx_signal_train, pilot_train, tx_bits_train)
        if torch.cuda.is_available():
            ModelInput1, ModelInput2, label = ModelInput1.cuda(), ModelInput2.cuda(), label.cuda()
        
        ModelOutput= Model(ModelInput1, ModelInput2)
        loss = criterion(ModelOutput, label)

        # ModelOutput, loss_dict = Model.get_loss(ModelInput1, ModelInput2, label)
        # loss = loss_dict['bit_loss'] + 1e-1 * loss_dict['signal_loss']
        
        predict = torch.where(ModelOutput >= 0, 1.0, 0.0)
        score = torch.where(predict == label, 1.0, 0.0)
        acc = torch.mean(score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Model Evaluating
        Model.eval()
        with torch.no_grad():
            val_ModelInput1, val_ModelInput2, val_label = generator(16, rx_signal_val, pilot_val, tx_bits_val)
            if torch.cuda.is_available():
                val_ModelInput1, val_ModelInput2, val_label = val_ModelInput1.cuda(), val_ModelInput2.cuda(), val_label.cuda()
            val_ModelOutput = Model(val_ModelInput1, val_ModelInput2)
            val_loss = criterion(val_ModelOutput, val_label).item()
            val_predict = torch.where(val_ModelOutput >= 0, 1.0, 0.0)
            val_score = torch.where(val_predict == val_label, 1.0, 0.0)
            val_acc = torch.mean(val_score)
            # print(f'Epoch: [{epoch}], Bit Loss {loss_dict["bit_loss"]:.4f}, Sig Loss {loss_dict["signal_loss"].item():.4f}, Acc {acc:.4f}, val_loss {val_loss:.4f}, val_acc {val_acc:.4f}')
            log_str = 'Epoch: [{0}]\t' 'Loss {loss:.4f}\t' 'Acc {acc:.4f}\t' 'val_loss {val_loss:.4f}\t' 'val_acc {val_acc:.4f}\t'.format(
                    epoch, loss=loss.item(), acc=acc, val_loss=val_loss, val_acc=val_acc)

            if epoch % 10 == 0:
                logger.info(log_str)
                writer.add_scalar('train/loss', loss.item(), epoch)
                writer.add_scalar('train/acc', acc, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/acc', val_acc, epoch)

            if val_loss < bestLoss and epoch % 100 == 0:
                # Model saving
                torch.save(Model, f'{model_save_dir}/receiver_{case}.pth.tar')
                logger.info(f"Best ever model saved, epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")
                bestLoss = val_loss

    logger.info("Best loss", bestLoss)
    torch.save(Model, f'{model_save_dir}/receiver_{case}_final.pth.tar')
    logger.info(f"Finval model saved, epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")
    logger.info(f'Training for case_{case} is finished!')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./../data')
    parser.add_argument('--case', type=str, default='1b')
    parser.add_argument('--max_epoch', type=int, default=20000)
    args = parser.parse_args()

    train(args.data_dir, args.case, args.max_epoch)
