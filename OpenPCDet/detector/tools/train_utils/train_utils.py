import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
# eval_loss用于评价验证集损失函数
# eval_mAP用于评价验证集mAP
#from tools.eval_utils.eval_utils import eval_mAP

def train_one_epoch(model, optimizer, train_loader,test_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch,total_it_each_epoch_val, cur_epoch,dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    
    #记录每一个epoch的损失
    epoch_loss=0
    tb_dict_epoch={'rpn_loss':0,'rcnn_loss':0,'point_loss_cls':0}
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        train_loss, tb_dict, disp_dict = model_func(model, batch)
        train_loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1
        disp_dict.update({'train_loss': train_loss.item(), 'lr': cur_lr})
        epoch_loss+=train_loss.item()
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                #tb_log.add_scalar('train/loss', train_loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    if key in tb_dict_epoch.keys():
                        #tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        tb_dict_epoch[key]+=val
    if rank == 0:
        pbar.close()
    train_epoch_loss=epoch_loss/total_it_each_epoch
    for key,val in tb_dict_epoch.items():
        tb_dict_epoch[key]=tb_dict_epoch[key]/total_it_each_epoch
    '''
    val_loss=eval_loss(model,test_loader,model_func=model_func,
                       total_it_each_epoch_val=total_it_each_epoch_val,accumulated_iter=accumulated_iter,
                       rank=rank,cur_epoch=cur_epoch,tb_log=tb_log)
    '''
    return accumulated_iter,train_epoch_loss,tb_dict_epoch


def train_model(model, optimizer, train_loader, test_loader,model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,choose_best=True,use_early_stopping=True):
    accumulated_iter = start_iter
    loss=float('inf')
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        total_it_each_epoch_val=len(test_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
            total_it_each_epoch_val=len(test_loader)// max(total_epochs,1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter,train_loss_of_cur_epoch,tb_dict_train= train_one_epoch(
                model, optimizer, train_loader,test_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                total_it_each_epoch_val=total_it_each_epoch_val,
                cur_epoch=cur_epoch,
                dataloader_iter=dataloader_iter
            )
            if tb_log is not None:
                tb_log.add_scalar('train/epoch_train_loss',train_loss_of_cur_epoch,cur_epoch)
                for key,val in tb_dict_train.items():
                    tb_log.add_scalar('train/'+key,val,cur_epoch)

            # save model with choose best，选取最好的模型保存，根据训练集上损失函数最小的保存
            if choose_best:
                trained_epoch = cur_epoch + 1
                best_ckpt_path=str(ckpt_save_dir / 'best')
                last_ckpt_path=str(ckpt_save_dir / 'last')
                # save best.pth
                if train_loss_of_cur_epoch<loss:
                    loss=train_loss_of_cur_epoch
                    save_checkpoint(
                            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=best_ckpt_path,
                        )
                else:
                    pass
                # save last.pth
                if trained_epoch==total_epochs:
                    save_checkpoint(
                            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=last_ckpt_path,
                        )
                else:
                    pass
            else:
            # save trained model without echoose best and early stopping
                trained_epoch = cur_epoch + 1
                if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                    ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                    ckpt_list.sort(key=os.path.getmtime)

                    if ckpt_list.__len__() >= max_ckpt_save_num:
                        for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                            os.remove(ckpt_list[cur_file_idx])

                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                    )
            # TODO:write early stopping here，not suggested
            if use_early_stopping:
                print("use early stopping")
            else:
                pass


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)