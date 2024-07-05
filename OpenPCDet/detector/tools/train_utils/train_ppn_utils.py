import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_


def ppn_train_one_epoch(model, optimizer, lr_scheduler,train_loader, test_loader, train_accumulated_iter,val_accumulated_iter,
                    rank, tbar, total_it_each_epoch, total_it_each_epoch_val, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    
    #记录每一个epoch的损失
    epoch_loss=0
    epoch_loss_backward=0
    tb_dict_epoch={'sample_loss':0, 'task_loss':0,'train_loss_backward':0}
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(train_accumulated_iter)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, train_accumulated_iter)
        model.train()
        optimizer.zero_grad()
        #forward函数调用位置
        # TODO:完善PointProposal中的forward函数
        # disp dict可能要代替
        train_loss, train_loss_backward, tb_dict, disp_dict = model(batch,is_training=True)
        # 使用 torchviz 可视化计算图
        # dot = make_dot(train_loss, params=dict(model.named_parameters()))
        # dot.render("simple_model_graph_wrong", format="png")
        #forward函数调用结束
        train_loss_backward.backward()
        clip_grad_norm_(model.parameters(), 10)
        #打印每个参数的梯度
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
        #     else:
        #         print("no grid")
        optimizer.step()
        train_accumulated_iter += 1
        disp_dict.update({'train_loss_backward': train_loss_backward.item(), 'train_loss':train_loss.item(), 'lr': cur_lr})
        epoch_loss+=train_loss.item()
        epoch_loss_backward+=train_loss_backward.item()
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=train_accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                #tb_log.add_scalar('train/loss', train_loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, train_accumulated_iter)
                for key, val in tb_dict.items():
                    if key in tb_dict_epoch.keys():
                        #tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        tb_dict_epoch[key]+=val
    if rank == 0:
        pbar.close()
    train_epoch_loss=epoch_loss/total_it_each_epoch
    train_epoch_loss_backward=epoch_loss_backward/total_it_each_epoch
    for key,val in tb_dict_epoch.items():
        tb_dict_epoch[key]=tb_dict_epoch[key]/total_it_each_epoch

    # val_accumulated_iter,val_loss,tb_dict_epoch_val=eval_loss(model, test_loader, val_accumulated_iter,
    #                 rank, tbar, total_it_each_epoch_val, dataloader_iter, tb_log=tb_log, leave_pbar=False)
    return train_accumulated_iter,train_epoch_loss,train_epoch_loss_backward, tb_dict_epoch

def eval_loss(model, test_loader, val_accumulated_iter,
                    rank, tbar, total_it_each_epoch_val, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch_val == len(test_loader):
        dataloader_iter = iter(test_loader)
    
    #记录每一个epoch的损失
    epoch_loss=0
    tb_dict_epoch={'sample_loss':0, 'task_loss':0}
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch_val, leave=leave_pbar, desc='val', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch_val):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(test_loader)
            batch = next(dataloader_iter)

        model.eval()
        #forward函数调用位置
        # TODO:完善PointProposal中的forward函数
        # disp dict可能要代替
        with torch.no_grad():
            val_loss, val_tb_dict, val_disp_dict = model(batch,is_training=True)
        #forward函数调用结束
        val_accumulated_iter += 1
        val_disp_dict.update({'val_loss': val_loss.item()})
        epoch_loss+=val_loss.item()
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=val_accumulated_iter))
            tbar.set_postfix(val_disp_dict)
            tbar.refresh()

            if tb_log is not None:
                #tb_log.add_scalar('train/loss', train_loss, accumulated_iter)
                for key, val in val_tb_dict.items():
                    if key in tb_dict_epoch.keys():
                        #tb_log.add_scalar('train/' + key, val, accumulated_iter)
                        tb_dict_epoch[key]+=val
    if rank == 0:
        pbar.close()
    val_epoch_loss=epoch_loss/total_it_each_epoch_val
    for key,val in tb_dict_epoch.items():
        tb_dict_epoch[key]=tb_dict_epoch[key]/total_it_each_epoch_val
    return val_accumulated_iter,val_epoch_loss,tb_dict_epoch

def ppn_train_model(model, optimizer, lr_scheduler, train_loader, test_loader, start_iter, start_epoch, total_epochs, 
                    train_sampler, rank, tb_log, ckpt_save_dir, choose_best=True):
    train_accumulated_iter = start_iter
    val_accumulated_iter = start_iter
    loss=float('inf')
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        total_it_each_epoch_val=len(test_loader)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)
            train_accumulated_iter,train_loss_of_cur_epoch, train_loss_of_cur_epoch_backward,tb_dict_train= ppn_train_one_epoch(
                model, optimizer,
                lr_scheduler=lr_scheduler,
                train_loader=train_loader,
                test_loader=test_loader,
                train_accumulated_iter=train_accumulated_iter,
                val_accumulated_iter=val_accumulated_iter,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                total_it_each_epoch_val=total_it_each_epoch_val,
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
                            checkpoint_state(model, optimizer, trained_epoch, val_accumulated_iter), filename=best_ckpt_path,
                        )
                else:
                    pass
                # save last.pth
                if trained_epoch==total_epochs:
                    save_checkpoint(
                            checkpoint_state(model, optimizer, trained_epoch, val_accumulated_iter), filename=last_ckpt_path,
                        )
                else:
                    pass
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