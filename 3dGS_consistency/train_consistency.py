import json
import os
import sys
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2

from utils_me import PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from my_camera_utils import cameraList_from_camInfos_myversion,CameraInfo
from typing import Dict, List, NamedTuple, Optional, Tuple,Union
import torch
from torch.utils.data import DataLoader, Dataset
from utils.image_utils import psnr, turbo_cmap
from utils.loss_utils import l1_loss, ssim
from tqdm import tqdm, trange

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn as nn
import torch.optim as optim
from scene.cameras import Camera

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

class CustomDataset(Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras 

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        # gt_image = self.cam_infos[idx].gt_image
        # input_image = self.cam_infos[idx].input_image
        # gt_image_tensor = PILtoTorch(gt_image, (800, 800))
        # input_image_tensor = PILtoTorch(input_image, (800, 800))

        gt_image_tensor=self.cameras[idx].original_image
        input_image_tensor=self.cameras[idx].input_image

        return input_image_tensor, gt_image_tensor
        
def readCamerasFromTransforms(
    path: str, transformsfile: str, white_background: bool, extension: str = ".png"
) -> List[CameraInfo]:
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        # print("cam_name", cam_name)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        # image_path = os.path.join(path, cam_name)
        image_path=cam_name
        image_name = Path(cam_name).stem
        # print("image_name", image_name)
        image = Image.open(image_path)

        #获取cam_name对应的dir路径
        cam_name_dir = os.path.dirname(cam_name)
        # print("cam_name_dir", cam_name_dir)
        gt_image_path = os.path.join(cam_name_dir, "rgba_bridge.png")
        input_image_path = os.path.join(cam_name_dir, "pbr_bridge.png")
        
        gt_image=Image.open(gt_image_path)
        input_image=Image.open(input_image_path)

        # im_data = np.array(image.convert("RGBA"))

        # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        # norm_data = im_data / 255.0
        # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        # image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
                gt_image=gt_image,
                input_image=input_image
            )
        )

    return cam_infos

def prepare_output_and_logger(args: GroupParams) -> Optional[SummaryWriter]:
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output_consistency/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    #添加log_dir path
    unique_str = str(uuid.uuid4())
    args.model_path=os.path.join(args.model_path, unique_str[0:10])

    print("Logdir path:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer: Optional[SummaryWriter],
    iteration: int,
    mse_loss: Union[float, torch.Tensor],
    testing_iterations: List[int],
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/mse_loss", mse_loss, iteration)


if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[10,50, 70, 100,150,200,280,350,400,600,800],
    )

    lp = ModelParams(parser)
    # parser.add_argument("--model_path", type=str, default="/home/jiahao/GS-IR/output_consistency",
    #                                         help="Path to the model")
    args = parser.parse_args(sys.argv[1:])
    lp.extract(args)

    print("data device:", lp.data_device)

    #设置tensorboard writer
    tb_writer = prepare_output_and_logger(args)

    path = "./datasets/TensoIR/lego/"
    print("reading training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background=False, extension=".png")
    print("reading testing Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background=False, extension=".png")

    # 采样测试代码 
    gt_image_sample = train_cam_infos[1].gt_image
    input_image_sample = train_cam_infos[1].input_image
    # 将gt_image_sample和input_image_sample保存到本地
    # gt_image_sample.save("gt_image_sample.png")
    # input_image_sample.save("input_image_sample.png")
    w,h=gt_image_sample.size
    resolution=(w,h)
    gt_image_tensor = PILtoTorch(gt_image_sample, resolution)
    print("gt_image tensor shape:",gt_image_tensor.shape)
   
    train_cameras=  cameraList_from_camInfos_myversion(train_cam_infos, 1.0, args)
    test_cameras=  cameraList_from_camInfos_myversion(test_cam_infos, 1.0, args)

    print("program is over")

    train_dataset = CustomDataset(train_cameras)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset= CustomDataset(test_cameras)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    model = UNet().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # output_dir = "./output_consistency"
    output_dir=args.model_path

    testing_iterations = args.test_iterations

    if args.eval:
        # Load the trained model
        model_path=os.path.join(output_dir, "unet_model.pth")
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}")
    else:

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        num_epochs = 800
        progress_bar = trange(0, num_epochs, desc="Training progress")  # For logging

        for epoch in range(num_epochs):
            model.train()

            iter_start.record()
            
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            iter_end.record()

            #program progress bar
            with torch.no_grad():
                loss=running_loss/len(train_loader)
                if(epoch%10==0):
                    progress_bar.set_postfix({"Loss":f"{loss:{.7}f}"})
                    progress_bar.update(10)
                if epoch==num_epochs:
                    progress_bar.close()
                                    
            # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
            # print(f"Epoch Time: {iter_start.elapsed_time(iter_end)/1000:.2f} s")

            # Tensorboard记录训练损失
            training_report(tb_writer, epoch, running_loss/len(train_loader),testing_iterations=testing_iterations)

            # 如果当前epoch在testing_iterations中，随机采样三张模型训练结果以及对应的gt图片
            if (epoch+1) in testing_iterations:
                model.eval()
                print(f"Testing at epoch {epoch}")
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(test_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()
                        outputs = model(inputs)

                        psnr_test=0.0
                        ssim_test=0.0
                        
                        for j in range(3):  # 采样三张图片
                            input_image = inputs[j].cpu().numpy().transpose(1, 2, 0)
                            input_image = (input_image * 255).astype(np.uint8)   #shape:(800,800,3)
                            print("input_image shape:", input_image.shape)
                            output_image = outputs[j].cpu().numpy().transpose(1, 2, 0)
                            output_image = (output_image * 255).astype(np.uint8) #shape:(800,800,3)
                            print("output_image shape:", output_image.shape)
                            gt_image = targets[j].cpu().numpy().transpose(1, 2, 0) #shape:(800,800,3)
                            gt_image = (gt_image * 255).astype(np.uint8)
                            print("gt_image shape:", gt_image.shape)
                            # 将图片保存到本地
                            if not os.path.exists(os.path.join(output_dir, "testing_result")):
                                os.makedirs(os.path.join(output_dir, "testing_result"))
                            input_pil_image = Image.fromarray(input_image)
                            input_pil_image.save(os.path.join(output_dir, f"testing_result/input_sample_epoch_{epoch+1}_img_{j}.png"))
                            output_pil_image = Image.fromarray(output_image)
                            output_pil_image.save(os.path.join(output_dir, f"testing_result/output_sample_epoch_{epoch+1}_img_{j}.png"))
                            gt_pil_image = Image.fromarray(gt_image)
                            gt_pil_image.save(os.path.join(output_dir, f"testing_result/gt_sample_epoch_{epoch+1}_img_{j}.png"))


                            # 计算PSNR和SSIM
                            image_tensor=torch.clamp(outputs[j],0,1)
                            gt_image_tensor=torch.clamp(targets[j],0,1)
                            psnr_test+=psnr(image_tensor, gt_image_tensor).mean().double()
                            ssim_test+=ssim(image_tensor, gt_image_tensor).mean().double()

                            # # 将图片添加到Tensorboard中
                            tb_writer.add_images(f"test/input_image", input_image, global_step=epoch, dataformats='HWC')
                            tb_writer.add_images(f"test/output_image", output_image, global_step=epoch, dataformats='HWC')
                            tb_writer.add_images(f"test/gt_image", gt_image, global_step=epoch, dataformats='HWC')
                            # tb_writer.add_image(f"input_sample_epoch_{epoch+1}_img_{j}", input_image, epoch, dataformats='HWC')
                            # tb_writer.add_image(f"output_sample_epoch_{epoch+1}_img_{j}", output_image, epoch, dataformats='HWC')
                            # tb_writer.add_image(f"gt_sample_epoch_{epoch+1}_img_{j}", gt_image, epoch, dataformats='HWC')

                        # 计算平均PSNR和SSIM
                        avg_psnr=psnr_test/3
                        avg_ssim=ssim_test/3
                        print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")

                        tb_writer.add_scalar(f"test_evaluate/avg_psnr", avg_psnr, epoch)
                        tb_writer.add_scalar(f"test_evaluate/avg_ssim", avg_ssim, epoch)

                        if i == 0:  # 只处理第一个batch
                            break
                # Sample from the training set
                model.eval()
                print(f"Sampling from training set at epoch {epoch}")
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()
                        outputs = model(inputs)

                        psnr_train=0.0
                        ssim_train=0.0
                        for j in range(3):  # Sample three images
                            input_image = inputs[j].cpu().numpy().transpose(1, 2, 0)
                            input_image = (input_image * 255).astype(np.uint8)
                            output_image = outputs[j].cpu().numpy().transpose(1, 2, 0)
                            output_image = (output_image * 255).astype(np.uint8)
                            gt_image = targets[j].cpu().numpy().transpose(1, 2, 0)
                            gt_image = (gt_image * 255).astype(np.uint8)

                            # Save images to local directory
                            if not os.path.exists(os.path.join(output_dir, "training_result")):
                                os.makedirs(os.path.join(output_dir, "training_result"))
                            input_pil_image = Image.fromarray(input_image)
                            input_pil_image.save(os.path.join(output_dir, f"training_result/train_input_sample_epoch_{epoch+1}_img_{j}.png"))
                            output_pil_image = Image.fromarray(output_image)
                            output_pil_image.save(os.path.join(output_dir, f"training_result/train_output_sample_epoch_{epoch+1}_img_{j}.png"))
                            gt_pil_image = Image.fromarray(gt_image)
                            gt_pil_image.save(os.path.join(output_dir, f"training_result/train_gt_sample_epoch_{epoch+1}_img_{j}.png"))

                            # 计算PSNR和SSIM
                            image_tensor=torch.clamp(outputs[j],0,1)
                            gt_image_tensor=torch.clamp(targets[j],0,1)
                            psnr_train+=psnr(image_tensor, gt_image_tensor).mean().double()
                            ssim_train+=ssim(image_tensor, gt_image_tensor).mean().double()

                            # Add images to Tensorboard
                            tb_writer.add_images(f"train/input_image", input_image, global_step=epoch, dataformats='HWC')
                            tb_writer.add_images(f"train/output_image", output_image, global_step=epoch, dataformats='HWC')
                            tb_writer.add_images(f"train/gt_image", gt_image, global_step=epoch, dataformats='HWC')
                            # tb_writer.add_image(f"train_input_sample_epoch_{epoch+1}_img_{j}", input_image, epoch, dataformats='HWC')
                            # tb_writer.add_image(f"train_output_sample_epoch_{epoch+1}_img_{j}", output_image, epoch, dataformats='HWC')
                            # tb_writer.add_image(f"train_gt_sample_epoch_{epoch+1}_img_{j}", gt_image, epoch, dataformats='HWC')

                        # 计算平均PSNR和SSIM
                        avg_psnr=psnr_train/3
                        avg_ssim=ssim_train/3
                        print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")

                        tb_writer.add_scalar(f"train_eval/avg_psnr", avg_psnr, epoch)
                        tb_writer.add_scalar(f"train_eval/avg_ssim", avg_ssim, epoch)

                        if i == 0:  # Only process the first batch
                            break
        
        # Save the trained model
        os.makedirs(output_dir, exist_ok=True)
        model_save_path=os.path.join(output_dir, "unet_model.pth")
        # model_save_path = "./output_consistency/unet_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    model.eval()

    count=0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            # Save the first image of the batch
            output_image = outputs[0].cpu().numpy().transpose(1, 2, 0)
            output_image = (output_image * 255).astype(np.uint8)
            output_pil_image = Image.fromarray(output_image)
            output_pil_image.save(os.path.join(output_dir, f"output_sample_{i}.png"))

            gt_image = targets[0].cpu().numpy().transpose(1, 2, 0)
            gt_image = (gt_image * 255).astype(np.uint8)
            gt_pil_image = Image.fromarray(gt_image)
            gt_pil_image.save(os.path.join(output_dir, f"gt_sample_{i}.png"))

            count+=1
            if count==10:
                break
            # if i == 0:  # Save only the first batch
            #     break
    
    
