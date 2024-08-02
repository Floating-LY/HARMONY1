from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _select_criterion_dis(self):
        criterion = nn.GaussianNLLLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,criterion_dis):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, level) in enumerate(vali_loader):
                level=level[0].to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]
                        else:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)
                else:
                    if self.args.output_attention:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]
                    else:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs,batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_dis =self._select_criterion_dis()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, level) in enumerate(train_loader):
                level=level[0].to(self.device)
                # print(batch_x.shape)
                # print(batch_x_mark.shape)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]
                        else:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mar,levels=level)
                    else:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, criterion_dis)
            test_loss = self.vali(test_data, test_loader, criterion, criterion_dis)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        means = []
        stds  = []
        preds_transed = []
        trues_transed = []
        means_transed = []
        stds_transed = []
        if self.args.NP:
           transformed_distributions_list=[]
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,level) in enumerate(test_loader):
                level=level[0].to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]
                        else:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)
                else:
                    if self.args.output_attention:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]

                    else:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                if self.args.NP:
                    transformed_distributions=[]
                    for _ in range(100):
                        z_sample = torch.randn(*mean.shape, device=mean.device) 
                        sample = z_sample * std + mean
                        transformed_output = self.model.real_nvp_transform(sample)
                        transformed_distributions.append(transformed_output)
                    transformed_distributions = torch.stack(transformed_distributions, dim=-1).detach().cpu().numpy()
                    transformed_distributions_list.append(transformed_distributions)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                mean = mean.detach().cpu().numpy()
                std = std.detach().cpu().numpy()
                preds_transed.append(outputs)
                trues_transed.append(batch_y)
                means_transed.append(mean)
                stds_transed.append(std)
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)
                    mean = test_data.inverse_transform(mean)
                    std = test_data.inverse_transform(std)
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                means.append(mean)
                stds.append(std)

        preds = np.array(preds)
        trues = np.array(trues)
        means = np.array(means)
        stds  = np.array(stds)
        preds_transed = np.array(preds_transed)
        trues_transed = np.array(trues_transed)
        means_transed = np.array(means_transed)
        stds_transed  = np.array(stds_transed)
        print('test shape:', preds.shape, trues.shape, means.shape, stds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        means = means.reshape(-1, means.shape[-2], trues.shape[-1])
        stds  = stds.reshape(-1, stds.shape[-2], stds.shape[-1])
        preds_transed = preds_transed.reshape(-1, preds_transed.shape[-2], preds_transed.shape[-1])
        trues_transed = trues_transed.reshape(-1, trues_transed.shape[-2], trues_transed.shape[-1])
        means_transed = means_transed.reshape(-1, means_transed.shape[-2], trues_transed.shape[-1])
        stds_transed  = stds_transed.reshape(-1, stds_transed.shape[-2], stds_transed.shape[-1])
        print('test shape:', preds.shape, trues.shape, means.shape, stds.shape)
        
        # result save
        if self.args.data=='Fisher':
            folder_path = './results/'+self.args.data_path + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        if self.args.NP:
            tds= np.array(transformed_distributions_list)
            tds  = tds.reshape(-1,tds.shape[-3], tds.shape[-2], tds.shape[-1])
            #print(tds.shape)
            mae, mse, rmse, mape, mspe, crps = metric(preds_transed, trues_transed,means_transed,stds_transed,tds)
            if self.args.data=='Fisher':
                np.save(folder_path + 'Samples.npy', tds)
        else:
            mae, mse, rmse, mape, mspe, crps = metric(preds_transed, trues_transed,means_transed,stds_transed,None)
        print('mse:{}, crps:{}'.format(mse,crps))
        f = open("result_"+str(self.args.data)+".txt", 'a')
        f.write(self.args.data_path +" " + self.args.model +" "+ str (self.args.seq_len) + "  \n")
        f.write('mse:{},crps:{}'.format(mse,crps))
        f.write('\n')
        f.write('\n')
        f.close()
        if self.args.data=='Fisher':
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,crps]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'means.npy', means)
            np.save(folder_path + 'stds.npy', stds)
            np.save(folder_path + 'true.npy', trues)
        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,level) in enumerate(pred_loader):
                level=level[0]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]
                        else:
                            outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)
                else:
                    if self.args.output_attention:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)[0]
                    else:
                        outputs,mean,std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,levels=level)

                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)
                means.append (mean)
                stds.append (stds)
        preds = np.array(preds)
        means = np.array(means)
        stds  = np.array(stds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/'+self.args.data_path + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        np.save(folder_path + 'real_means.npy', means)
        np.save(folder_path + 'real_stds.npy', stds)
        return