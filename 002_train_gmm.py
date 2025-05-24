import tqdm
import torch
import config
import datasets
import tools.utils as utils
import torch.nn.functional as F
from tools.logging import Logger
from tools.loss import LossOperator
from einops import rearrange
from models.model_gmm import CAFWM


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.stage = 'GMM'

        self.args = config.MetricsInit(self.args)
        self.logger = Logger(self.args, self.args.SaveFolder_GMM)
        self.lossOp = LossOperator(self.args)
        self.visualizer = utils.Visualizer(self.args, self.args.SaveFolder_GMM)
        self._build_model()
        self._get_dataloader()
        self._set_optimizer()
    
    def _set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr, betas=[self.args.beta1, 0.999])
    
    def _get_dataloader(self):
        train_dataset = datasets.ImagesDataset(self.args, phase='train')
        self.train_dataLoader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.BatchSize, shuffle=True, persistent_workers=True, pin_memory=True, num_workers=self.args.NumWorkers)
        self.args.len_dataset = len(self.train_dataLoader)

    def _build_model(self):
        self.net = CAFWM(self.args).to(self.args.device)
        self.logger.print_network(self.net, 'net')
        utils.weights_initialization(self.net, 'xavier', gain=1.0)

    def _set_mode(self, is_train=True):
        self.net.train() if is_train else self.net.eval()

    def _visualize(self, vis_list, order):
        de_norm = lambda x: (x + 1) / 2 * 255.
        vis_list = [de_norm(image).detach().cpu() for image in vis_list]
        self.visualizer.vis_train(vis_list, order)
                           
    def _train_one_epoch(self):
        self._set_mode()
        num_layers = 4
        with tqdm.tqdm(self.train_dataLoader, desc="training") as pbar:
            for idx, sample in enumerate(pbar):
                image       = sample['image'].to(self.args.device)
                cloth       = sample['cloth'].to(self.args.device)
                cloth_mask  = sample['cloth_mask'].to(self.args.device)
                image_pose  = sample['image_pose'].to(self.args.device)
                agnostic    = sample['agnostic'].to(self.args.device)
                person_shape= sample['person_shape'].to(self.args.device)
                person_clothes = sample['person_clothes'].to(self.args.device)
                person_clothes_mask = sample['person_clothes_mask'].to(self.args.device)
                image_grid  = sample['image_grid'].to(self.args.device)

                loss_warp = 0
                loss_mask = 0
                loss_styles = 0
                loss_smooth = 0
                loss_contents = 0
                loss_laplacian = 0
                with torch.set_grad_enabled(True):
                    output = self.net(cloth, cloth_mask, person_shape)
                    for i in range(num_layers):
                        warping_mask    = output['warping_masks'][i]
                        warping_cloth   = output['warping_cloths'][i]
                        shape_last_flow = output['shape_last_flows'][i]
                        cloth_last_flow = output['cloth_last_flows'][i]
                        shape_delta_flow = output['shape_delta_flows'][i]
                        cloth_delta_flow = output['cloth_delta_flows'][i]

                        _, _, h, w = warping_mask.shape
                        person_clothes_ = F.interpolate(person_clothes, size=(h, w))
                        person_clothes_mask_ = F.interpolate(person_clothes_mask, size=(h, w))

                        loss_content, loss_style = self.lossOp.calc_vgg_loss(warping_cloth, person_clothes_)
                        loss_warp     += self.lossOp.criterion_L1(warping_cloth, person_clothes_) * (i+1)
                        loss_mask     += self.lossOp.criterion_L1(warping_mask, person_clothes_mask_) * (i+1)
                        loss_laplacian+= (self.lossOp.calc_laplacian_loss(shape_last_flow) + self.lossOp.calc_laplacian_loss(cloth_last_flow)) / 2 * (i+1) * 6
                        loss_smooth   += (self.lossOp.calc_total_variation_loss(shape_delta_flow) + self.lossOp.calc_total_variation_loss(cloth_delta_flow)) / 2 * 0.1
                        loss_contents += loss_content * (i+1) * 0.2
                        loss_styles   += loss_style * (i+1) * 10
                    loss = loss_contents + loss_styles + loss_laplacian + loss_smooth + loss_mask + loss_warp

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if idx % 1000 == 0:
                    warped_cloth = output['warping_cloths'][-1]
                    cloth_last_flows = output['cloth_last_flows'][-1]

                    warped_grid = self.net.dec_tryon.spatial_transform(image_grid, cloth_last_flows, padding_mode='zeros')
                    self._visualize([image_pose, agnostic], order=1) 
                    self._visualize([cloth, warped_cloth, person_clothes], order=2)    
                    self._visualize([warped_grid, warped_cloth, (warped_cloth + image)*0.5], order=3)
                self.logger.loss_tmp['wloss-content'] += loss_contents.item()
                self.logger.loss_tmp['wloss-mask']    += loss_mask.item()
                self.logger.loss_tmp['wloss-style']   += loss_styles.item()
                self.logger.loss_tmp['wloss-lapla']   += loss_laplacian.item()
                self.logger.loss_tmp['wloss-smooth']  += loss_smooth.item()
                self.logger.loss_tmp['wloss-warp']    += loss_warp.item()
                self.logger.loss_tmp['loss-total']   += loss.item()
                pbar.set_description(f"Epoch: {self.args.epoch}, Loss: {self.logger.loss_tmp['loss-total'] / (idx + 1):.4f}")
            self.logger.loss_tmp = {k : v / self.args.len_dataset for k, v in self.logger.loss_tmp.items()}
        return self.logger.loss_tmp['loss-total']


    def train(self):
        print(self.args)
        for epoch in range(0, self.args.epochs):
            utils.flush()
            self.args.epoch = epoch + 1

            loss = self._train_one_epoch()
            
            self.logger.Log_PerEpochLoss()
            self.logger.Reset_PerEpochLoss()
            self.visualizer.plot_loss(self.logger.loss_history)

            utils.save_checkpoint(self.args.RootCheckpoint_GMM, {
                'args': self.args, 'state_dict': self.net.state_dict(),
            }, loss < self.args.best_loss)

            self.args.best_loss = min(loss, self.args.best_loss)


if __name__ == '__main__':
    trainer = Trainer(config.GetConfig())
    trainer.train()

