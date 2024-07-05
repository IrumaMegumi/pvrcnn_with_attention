# Point Proposal Network, source points are painted
# 建议分阶段独立训练
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable
import torch.distributions as dist
import time
import pdb

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=8)
        self.conv1 = torch.nn.Conv1d(8, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)
        
class TopKBinaryMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, k):
        # 保存输入和k以备反向传播使用
        ctx.save_for_backward(input_tensor)
        ctx.k = k
        
        # 对输入张量进行排序
        sorted_tensor, indices = torch.sort(input_tensor[:, 0], descending=True)
        
        # 创建编码张量
        encoded_tensor = torch.zeros_like(input_tensor)
        encoded_tensor[indices[:k]] = 1
        return encoded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        k = ctx.k
        
        # 创建一个全零的梯度张量
        grad_input = torch.zeros_like(input_tensor)
        
        # 对输入张量进行排序
        _, indices = torch.sort(input_tensor, descending=True)
        
        # 将梯度传递给前k个元素
        grad_input[indices[:k]] = grad_output[indices[:k]]
        grad_input[indices[k:]]=grad_output[indices[k:]]
        return grad_input, None
    
class TopKBinaryMaskFunction_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, k):
        # 保存输入和k以备反向传播使用
        ctx.save_for_backward(input_tensor)
        ctx.k = k
        
        # 对输入张量进行排序
        sorted_tensor, indices = torch.sort(input_tensor[:, 0], descending=True)
        
        # 创建编码张量
        encoded_tensor = torch.zeros_like(input_tensor)
        encoded_tensor[indices[:k]] = 1
        return encoded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        k = ctx.k

        # 创建一个全零的梯度张量
        grad_input = torch.zeros_like(input_tensor)
        
        # 对输入张量进行排序
        sorted_tensor, indices = torch.sort(input_tensor[:, 0], descending=True)
        # 计算前k个元素的梯度
        grad_input[indices[:k]] = grad_output[indices[:k]]
        # 计算后k个元素的梯度, 使用sigmoid作为连续化方法
        grad_input[indices[k:]] = torch.sigmoid(input_tensor[indices[k:]])
        return grad_input, None
#关键点建议网络，注意Loss的处理
#可能需要做好多阶段训练的准备
class PointProposalNet(nn.Module):
    '''
        num_object_points: 你j从原始点云中初筛的点数量
        num_keypoints: 采样关键>点数量
    '''
    def __init__(self,num_object_points,num_keypoints,global_feat=False):
        super(PointProposalNet,self).__init__()
        self.feature_extraction_module=PointNetfeat(global_feat=global_feat)
        self.num_object_points=num_object_points
        self.num_keypoints=num_keypoints
        if self.num_object_points<self.num_keypoints:
            raise ValueError(f"num object points should be more than num keypoints,but your num object points is {self.num_object_points}, your num key points is {self.num_keypoints}")
        else:
            pass
        self.W_gate = nn.Linear(1088, 1088)
        self.b_gate = nn.Parameter(torch.zeros(1088))
        self.W_fc = nn.Linear(1088, 1088)
        #门控函数
        self.gate_function=nn.Sigmoid()
        
        self.evaluate_layer_1=nn.Linear(1088,512)
        self.instanceNorm_1=nn.InstanceNorm1d(512)
        self.evaluate_layer_2=nn.Linear(512,256)
        self.instanceNorm_2=nn.InstanceNorm1d(256)
        self.evaluate_layer_3=nn.Linear(256,1)
        self.dropout = nn.Dropout(p=0.3)
        self.activate_func_eval=nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.core_sample_loss=nn.SmoothL1Loss()
    def forward(self,batch_dict,is_training=False):
        #input shape:[batch(should be 1), num_points, num_point_features+1]
        #TODO: 对painted points的squeeze放在外面，我不太建议放在里面
        '''
        Attention: 所有的数据不带有任何梯度跟踪效果，目前存在跟踪错误
        TODO: 跟踪排查
        TODO:适配ppn训练和PV-RCNN训练注释
        '''
        painted_points=batch_dict['painted_points']
        #painted_points=torch.from_numpy(batch_dict['painted_points']) # train_ppn
        batch_size=batch_dict['batch_size']
        #当前batch的损失值总和
        train_loss_batch=0
        tb_dict_batch={'train_loss':0, 'sample_loss':0, 'task_loss': 0}
        #keypoints_list=[] #train_ppn
        #当前batch中每个图片关键点的提取和损失的计算
        for cur_batch in range(batch_size):
            cur_painted_points=painted_points[painted_points[:, 0] == cur_batch]
            cur_painted_points=cur_painted_points[:, 1:]
            _, indices = torch.sort(cur_painted_points[:, 4])
            sorted_points = cur_painted_points[indices]
            object_points= sorted_points[:self.num_object_points, :]
            #object_points为初步筛选后的点，数量为6000(未排序)
            object_points= object_points.unsqueeze(0)
            object_points= object_points.permute(0,2,1)
            object_points= object_points.to(self.device)
            object_points= object_points.to(torch.float32)
            object_point_features=self.feature_extraction_module(object_points)
            object_point_features=object_point_features.permute(0,2,1)
            #x to context fusion:[batch,num_points,num_point_features]
            #TODO: don't forget to concat it to the original point features
            # 调制特征
            g = self.gate_function(self.W_gate(object_point_features) + self.b_gate)
            # 上下文门控特征
            result_features = g * self.W_fc(object_point_features)
            scores_1=self.activate_func_eval(self.instanceNorm_1(self.evaluate_layer_1(result_features)))
            scores_2=self.activate_func_eval(self.instanceNorm_2(self.dropout(self.evaluate_layer_2(scores_1))))
            scores_final=self.evaluate_layer_3(scores_2)#[N,1]
            scores_final=scores_final.squeeze(0)
            #scores_final: 神经网络输出后的得分
            #object_points_with_scores: 带有得分的点，维度为6000，最后一列是score
            #scores: 排序后的得分
            #indices:排序后得分从大到小的索引
            scores, indices = torch.sort(scores_final[:, 0], descending=True)
            keypoints_scores=scores[:self.num_keypoints].unsqueeze(1)
            # post processing, 注意：scores带梯度，建议和keypoints做一个级联处理
            object_points=object_points.permute(0,2,1).squeeze(0)
            #排序后的6000个点
            sorted_points = object_points[indices]
            #取前2048个点作为关键点
            keypoints= sorted_points[:self.num_keypoints, :3]           
            if is_training==True:
                scores_final=TopKBinaryMaskFunction_v2.apply(scores_final,self.num_keypoints)
                # scores_final[indices[:self.num_keypoints]]=1
                # scores_final[indices[self.num_keypoints:]]=0
                #采用原object_points和编码后的scores_final级联，使采样操作连续
                encoded_keypoints=torch.cat((object_points[:,:3],scores_final),dim=1)
                train_loss_cur_batch, tb_dict_cur_batch, disp_dict=self.calculate_loss(encoded_keypoints,object_points[:,:3], scores_final, batch_dict, cur_batch)
                for key,val in tb_dict_cur_batch.items():
                    tb_dict_batch[key]+=val 
                train_loss_batch+=train_loss_cur_batch
            else:
                #batch_indices_tensor=torch.full((self.num_keypoints,1),cur_batch).to(self.device) #train_ppn
                #keypoints=torch.cat((batch_indices_tensor,keypoints),dim=1) #train_ppn
                #keypoints_list.append(keypoints) #train_ppn
                pass
        if is_training==True:
            #训练完这个batch的所有点云后把损失做平均值返回
            for key,val in tb_dict_batch.items():
                tb_dict_batch[key]=tb_dict_batch[key]/batch_size
            train_loss_batch=train_loss_batch/batch_size
            return train_loss_batch, tb_dict_batch, disp_dict
        #返回采样点和采样时提取的原始点特征，后续不确定是否备用
        #keypoints=torch.cat(keypoints_list,dim=0)  
        return keypoints
    
    #先写好训练的代码，然后再写loss
    def calculate_loss(self, encoded_keypoints,object_points, scores_final,batch_dict,cur_batch):
        disp_dict={}
        sample_loss, tb_dict=self.calculate_sample_loss(encoded_keypoints, batch_dict,cur_batch)
        task_loss, tb_dict =self.calculate_task_loss(object_points, scores_final,batch_dict, tb_dict, cur_batch)
        loss= 0.7*task_loss+0.3*sample_loss
        tb_dict.update({'train_loss':loss})
        return loss, tb_dict, disp_dict

    #点到各个框中心最小的smooth_l1 loss总和除以采样点总数，计算的是当前batch中的一张图
    def calculate_sample_loss(self, encoded_keypoints, batch_dict, cur_batch):
        '''
        input: keypoints: 当前batch的一张图采样的关键点
        batch_dict: 当前batch的所有数据
        cur_batch: 你想处理当前batch的第几组数据
        我有点懒，下面所有的Keypoints可以理解为encoded keypoints
        '''
        gt_boxes_center_cur_batch=torch.from_numpy(batch_dict['gt_boxes'][cur_batch,:,:3]).to(self.device) #当前batch中gt_boxes的中心坐标
        mask = gt_boxes_center_cur_batch.sum(dim=1) != 0
        filtered_gt_boxes = gt_boxes_center_cur_batch[mask] #过滤掉全部为0的box
        keypoints_coord=encoded_keypoints[:,:3]
        keypoints_is_sampled=encoded_keypoints[:,-1].unsqueeze(1)
        num_gt_boxes=filtered_gt_boxes.shape[0]
        num_keypoints=encoded_keypoints.shape[0]
        # 计算所有关键点与所有gt box中心之间的L1距离
        expanded_keypoints = keypoints_coord.unsqueeze(1).expand(-1, filtered_gt_boxes.size(0), -1)
        expanded_gt_boxes = filtered_gt_boxes.unsqueeze(0).expand(keypoints_coord.size(0), -1, -1)
        loss_matrix = F.smooth_l1_loss(expanded_keypoints, expanded_gt_boxes,reduction='none').sum(dim=2)
        loss_matrix=loss_matrix*keypoints_is_sampled
        min_losses, _ = loss_matrix.min(dim=1)
        avg_min_loss = min_losses.sum()/self.num_keypoints
        tb_dict={'sample_loss': avg_min_loss}
        return avg_min_loss, tb_dict
    
    #近远处点的比例和标准比例差的绝对值
    def calculate_task_loss(self, object_points, scores_final, batch_dict,tb_dict, cur_batch):
        '''
        sorted_object_points:[6000个点特征，score]
        '''
        is_near_cur_batch=torch.from_numpy(batch_dict['near'][cur_batch,:,:]).to(self.device).to(torch.float32)
        mask =is_near_cur_batch.sum(dim=1) != 0
        filtered_is_near = is_near_cur_batch[mask] #过滤掉全部为0的box
        def threshold_function(tensor, value, epsilon=1e-3):
            return torch.exp(-((tensor - value).abs() / epsilon))
        count_1 = torch.sum(filtered_is_near == 1).item()
        count_2 = torch.sum(filtered_is_near == 2).item()
        #损失函数计算的比例：近处点数量/远处点数量 应当接近ground truth中近远处框的比例
        if count_1!=0 and count_2!=0:
            ratio_gtbox_near_far=count_1/count_2
        elif count_1!=0 and count_2==0:
            ratio_gtbox_near_far=count_1
        elif count_1==0 and count_2!=0:
            ratio_gtbox_near_far=1/count_2
        else:
            ratio_gtbox_near_far=1
        # ratio_gtbox_near_far=count_1/(count_2+0.01)
        near_far_tensor=torch.where(
            (torch.abs(object_points[:, 0]) < 35.2) & (torch.abs(object_points[:, 1]) < 20),
            torch.tensor(1.0,device=self.device,dtype=torch.float32,requires_grad=True),
            torch.tensor(2.0,device=self.device,dtype=torch.float32,requires_grad=True)
        ).unsqueeze(1)
        near_far_tensor=scores_final*near_far_tensor
        #损失函数计算的比例：近处点数量/远处点数量 应当接近ground truth中近远处框的比例
        keypoint_near = threshold_function(near_far_tensor, 1).sum()
        keypoint_far = threshold_function(near_far_tensor, 2).sum()
        ratio_keypoint_near_far=keypoint_near/(keypoint_far+1)

        task_loss=torch.abs(torch.log10(torch.abs(ratio_gtbox_near_far-ratio_keypoint_near_far)))
        tb_dict.update({'task_loss': task_loss})
        return task_loss, tb_dict   

class PointProposalNet_v2(nn.Module):
    '''
        num_object_points: 你从原始点云中初筛的点数量
        num_keypoints: 采样关键>点数量
    '''
    def __init__(self,num_object_points,num_keypoints,global_feat=False):
        super(PointProposalNet_v2,self).__init__()
        self.feature_extraction_module=PointNetfeat(global_feat=global_feat)
        self.num_object_points=num_object_points
        self.num_keypoints=num_keypoints
        if self.num_object_points<self.num_keypoints:
            raise ValueError(f"num object points should be more than num keypoints,but your num object points is {self.num_object_points}, your num key points is {self.num_keypoints}")
        else:
            pass
        self.W_gate = nn.Linear(1088, 1088)
        self.b_gate = nn.Parameter(torch.zeros(1088))
        self.W_fc = nn.Linear(1088, 1088)
        #门控函数
        self.gate_function=nn.Sigmoid()
        
        self.evaluate_layer_1=nn.Linear(1088,512)
        self.instanceNorm_1=nn.InstanceNorm1d(512)
        self.evaluate_layer_2=nn.Linear(512,256)
        self.instanceNorm_2=nn.InstanceNorm1d(256)
        self.evaluate_layer_3=nn.Linear(256,1)
        self.dropout = nn.Dropout(p=0.3)
        self.activate_func_eval=nn.ReLU()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.core_sample_loss=nn.SmoothL1Loss()
    def forward(self,batch_dict,is_training=False):
        #input shape:[batch(should be 1), num_points, num_point_features+1]
        #TODO: 对painted points的squeeze放在外面，我不太建议放在里面
        '''
        Attention: 所有的数据不带有任何梯度跟踪效果，目前存在跟踪错误
        TODO: 跟踪排查
        '''
        painted_points=batch_dict['painted_points']
        #painted_points=torch.from_numpy(batch_dict['painted_points']) #used for training ppn
        batch_size=batch_dict['batch_size']
        #当前batch的损失值总和
        train_loss_batch=0
        tb_dict_batch={'train_loss':0, 'sample_loss':0, 'task_loss': 0,'train_loss_backward':0}
        keypoints_list=[] #train_ppn
        #当前batch中每个图片关键点的提取和损失的计算
        for cur_batch in range(batch_size):
            cur_painted_points=painted_points[painted_points[:, 0] == cur_batch]
            cur_painted_points=cur_painted_points[:, 1:]
            _, indices = torch.sort(cur_painted_points[:, 4])
            sorted_points = cur_painted_points[indices]
            object_points= sorted_points[:self.num_object_points, :]
            #object_points为初步筛选后的点，数量为6000(未排序)
            object_points= object_points.unsqueeze(0)
            object_points= object_points.permute(0,2,1)
            object_points= object_points.to(self.device)
            object_points= object_points.to(torch.float32)
            object_point_features=self.feature_extraction_module(object_points)
            object_point_features=object_point_features.permute(0,2,1)
            #x to context fusion:[batch,num_points,num_point_features]
            #TODO: don't forget to concat it to the original point features
            # 调制特征
            g = self.gate_function(self.W_gate(object_point_features) + self.b_gate)
            # 上下文门控特征
            result_features = g * self.W_fc(object_point_features)
            scores_1=self.activate_func_eval(self.instanceNorm_1(self.evaluate_layer_1(result_features)))
            scores_2=self.activate_func_eval(self.instanceNorm_2(self.dropout(self.evaluate_layer_2(scores_1))))
            scores_final=self.evaluate_layer_3(scores_2)
            scores_final=scores_final.squeeze(0)
            #object_points_with_scores: 带有得分的点，维度为6000，最后一列是score
            #scores: 排序后的得分
            #indices:排序后得分从大到小的索引
            object_points=object_points.permute(0,2,1).squeeze(0)
            objpoints_with_scores=torch.cat((object_points,scores_final),dim=1) #未排序前的点和未排序得分的级联
            scores, indices = torch.sort(objpoints_with_scores[:, -1], descending=True)
            #keypoints_scores: 关键点得分
            keypoints_scores=scores[:self.num_keypoints].unsqueeze(1)
            # TODO:scores_final用Torch.distributed算概率分布，使用log_prob算对数概率
            #排序后的6000个点
            sorted_points = object_points[indices]
            #取前2048个点作为关键点
            keypoints= sorted_points[:self.num_keypoints, :3]
            scores_final=torch.sigmoid(scores_final)
            bernoulli_dist = dist.Bernoulli(probs=scores_final)
            log_prob = bernoulli_dist.log_prob(torch.ones_like(scores_final))
            if is_training==True:
                #采用原object_points和编码后的scores_final级联，使采样操作连续
                train_loss_cur_batch, train_loss_backward, tb_dict_cur_batch, disp_dict=self.calculate_loss(keypoints, log_prob, batch_dict, cur_batch)
                for key,val in tb_dict_cur_batch.items():
                    tb_dict_batch[key]+=val 
                train_loss_batch+=train_loss_cur_batch
            else:
                # batch_indices_tensor=torch.full((self.num_keypoints,1),cur_batch).to(self.device) 
                # keypoints=torch.cat((batch_indices_tensor,keypoints),dim=1)
                # keypoints_list.append(keypoints) #train_ppn
                pass
        if is_training==True:
            #训练完这个batch的所有点云后把损失做平均值返回
            for key,val in tb_dict_batch.items():
                tb_dict_batch[key]=tb_dict_batch[key]/batch_size
            train_loss_batch=train_loss_batch/batch_size
            train_loss_backward=train_loss_backward/batch_size
            return train_loss_batch, train_loss_backward, tb_dict_batch, disp_dict
        #返回采样点和采样时提取的原始点特征，后续不确定是否备用
        #keypoints=torch.cat(keypoints_list,dim=0)  #train_ppn
        return keypoints
    
    #先写好训练的代码，然后再写loss
    def calculate_loss(self, keypoints, log_prob, batch_dict,cur_batch):
        '''
        input:
            keypoints:采样后的关键点xyz坐标
            scores_final:神经网络输出的得分
        '''
        disp_dict={}
        sample_loss, tb_dict=self.calculate_sample_loss(keypoints, batch_dict, cur_batch)
        task_loss, tb_dict =self.calculate_task_loss(keypoints, batch_dict, tb_dict, cur_batch)
        loss= 0.7*task_loss+0.3*sample_loss
        #TODO:乘概率密度算平均值
        loss_backward=torch.mean(log_prob*loss)
        tb_dict.update({'train_loss':loss.item(),'train_loss_backward':loss_backward.item()})
        return loss,loss_backward, tb_dict, disp_dict

    #点到各个框中心最小的smooth_l1 loss总和除以采样点总数，计算的是当前batch中的一张图
    def calculate_sample_loss(self, keypoints, batch_dict, cur_batch):
        '''
        input: keypoints: 当前batch的一张图采样的关键点
        batch_dict: 当前batch的所有数据
        cur_batch: 你想处理当前batch的第几组数据
        keypoints:关键点，带xyz
        '''
        gt_boxes_center_cur_batch=torch.from_numpy(batch_dict['gt_boxes'][cur_batch,:,:3]).to(self.device) #当前batch中gt_boxes的中心坐标
        mask = gt_boxes_center_cur_batch.sum(dim=1) != 0
        filtered_gt_boxes = gt_boxes_center_cur_batch[mask] #过滤掉全部为0的box
        # 计算所有关键点与所有gt box中心之间的L1距离
        expanded_keypoints = keypoints.unsqueeze(1).expand(-1, filtered_gt_boxes.size(0), -1)
        expanded_gt_boxes = filtered_gt_boxes.unsqueeze(0).expand(keypoints.size(0), -1, -1)
        loss_matrix = F.smooth_l1_loss(expanded_keypoints, expanded_gt_boxes,reduction='none')
        min_losses, _ = loss_matrix.min(dim=1)
        avg_min_loss = min_losses.sum()/self.num_keypoints
        tb_dict={'sample_loss': avg_min_loss}
        return avg_min_loss, tb_dict
    
    #近远处点的比例和标准比例差的绝对值
    def calculate_task_loss(self, keypoints, batch_dict,tb_dict, cur_batch):
        '''
        sorted_object_points:[6000个点特征，score]
        '''
        is_near_cur_batch=torch.from_numpy(batch_dict['near'][cur_batch,:,:]).to(self.device).to(torch.float32)
        mask =is_near_cur_batch.sum(dim=1) != 0
        filtered_is_near = is_near_cur_batch[mask] #过滤掉全部为0的box
        def threshold_function(tensor, value, epsilon=1e-3):
            return torch.exp(-((tensor - value).abs() / epsilon))
        count_1 = torch.sum(filtered_is_near == 1).item()
        count_2 = torch.sum(filtered_is_near == 2).item()
        #损失函数计算的比例：近处点数量/远处点数量 应当接近ground truth中近远处框的比例
        if count_1!=0 and count_2!=0:
            ratio_gtbox_near_far=count_1/count_2
        elif count_1!=0 and count_2==0:
            ratio_gtbox_near_far=count_1
        elif count_1==0 and count_2!=0:
            ratio_gtbox_near_far=1/count_2
        else:
            ratio_gtbox_near_far=1
        # ratio_gtbox_near_far=count_1/(count_2+0.01)
        near_far_tensor=torch.where(
            (torch.abs(keypoints[:, 0]) < 35.2) & (torch.abs(keypoints[:, 1]) < 20),
            torch.tensor(1.0,device=self.device,dtype=torch.float32,requires_grad=True),
            torch.tensor(2.0,device=self.device,dtype=torch.float32,requires_grad=True)
        ).unsqueeze(1)
        #损失函数计算的比例：近处点数量/远处点数量 应当接近ground truth中近远处框的比例
        keypoint_near = threshold_function(near_far_tensor, 1).sum()
        keypoint_far = threshold_function(near_far_tensor, 2).sum()
        ratio_keypoint_near_far=keypoint_near/(keypoint_far+1)
        ratio_loss=ratio_gtbox_near_far/ratio_keypoint_near_far
        if ratio_loss<1e-3:
            ratio_loss=torch.tensor(1e-3,dtype=torch.float32).to(self.device)
        elif ratio_loss>1000:
            ratio_loss=torch.tensor(1000,dtype=torch.float32).to(self.device)
        task_loss=torch.abs(torch.log10(torch.abs(ratio_loss)))
        tb_dict.update({'task_loss': task_loss})
        return task_loss, tb_dict  
    
if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pointfeat = PointProposalNet(num_object_points=6000,num_keypoints=2048,global_feat=False)
    pointfeat=pointfeat.to(device)
    while(1):
        sim_data = torch.rand(1,30000,9)
        sim_data=sim_data.to(device)
        start=time.time()
        keypoints,keypointfeat= pointfeat(sim_data, is_training=True)
        end=time.time()
        run_time=(end-start)*1000
        print('point feat', keypoints.size())
        print(f"{run_time}ms")