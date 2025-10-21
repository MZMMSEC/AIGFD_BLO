import pdb
import torch, os, random, time, sys, argparse, datetime, pickle
import numpy as np
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from model import CLIP_EXIF_Net
from dataset.SSL_training_dataset import selectiveEXIF_data
from CLIP import clip
from logger import create_logger
from loss import L2R_Loss, Fidelity_Loss_multi, categorical_focal_loss_fidelity
from BLO import BL_Optimizer
from utils import get_weight_str, create_task_flags_sperate, predict_batch, accuracy_batch


DEVICE= "cuda" if torch.cuda.is_available() else "cpu"

def collect_separate_texts_v2(num_exif_scale=3):
    if num_exif_scale == 3:
        exif_scale = ['low', 'medium', 'high'] # 从左到右对应的提取数值越大
    else:
        exif_scale = ['ultra-low', 'low', 'medium', 'high', 'ultra-high']
    # faces = ['regular', 'irregular']
    faces = ['photographic', 'manipulated']
    # locals = ['mouth', 'eye']
    locals = ['mouth', 'eye', 'nose']

    makes = ['Nikon', 'Apple', 'FUJIFILM', 'Canon', 'Samsung', 'Sony', 'Panasonic', 'Nokia', 'Pentax', 'others'] # 21
    metering_mode = ['average', 'centerweighted', 'others', 'partial', 'spot', 'multi-spot', 'multi-segment', 'evaluative'] # 29
    exposure_mode = ['auto', 'auto-bracketing', 'program', 'manual', 'aperture-priority', 'easy-shooting', 'shutter-priority'] # 36
    white_balance_mode = ['auto', 'manual'] # 38
    exposure_program = ['landscape mode', 'manual control', 'normal program', 'portrait mode', 'shutter priority', 'aperture priority'] # 44

    # for EXIF tags templates
    iso_texts = [f"a photo of a face with the {iso} value of ISO speed rating" for iso in exif_scale]
    av_texts = [f"a photo of a face with the {av} value of aperture" for av in exif_scale]
    et_texts = [f"a photo of a face with the {et} value of exposure time" for et in exif_scale]
    fl_texts = [f"a photo of a face with the {fl} value of focal length" for fl in exif_scale]
    ## for categorical
    make_texts = [f"a photo of a face taken with the {make} brand camera" for make in makes]
    metering_mode_texts = [f"A photo of the face captured using the {mm} metering mode" for mm in metering_mode]
    exposure_mode_texts = [f"A photo of the face captured using the {em} exposure mode" for em in exposure_mode]
    white_balance_mode_texts = [f"A photo of the face captured using the {wb} white balance mode" for wb in white_balance_mode]
    exposure_program_mode_texts = [f"A photo of the face captured using the exposure program set to {ep}" for ep in exposure_program]

    # for face types templates
    face_texts_global = [f"a photo of a {face_type} face" for face_type in faces]
    # face_texts_local = [f"a photo of a face with an irregular {local_face} region" for local_face in locals]
    face_texts_local = [f"a photo of a face with a manipulated {local_face} region" for local_face in locals]

    # # for binary classification templates
    # bc_texts = ["a photographic face image", "an AI-generated face image"]

    cat_texts_all = []
    cat_texts_all.extend(iso_texts)
    cat_texts_all.extend(av_texts)
    cat_texts_all.extend(et_texts)
    cat_texts_all.extend(fl_texts)
    cat_texts_all.extend(make_texts)
    cat_texts_all.extend(metering_mode_texts)
    cat_texts_all.extend(exposure_mode_texts)
    cat_texts_all.extend(white_balance_mode_texts)
    cat_texts_all.extend(exposure_program_mode_texts)

    cat_texts_all.extend(face_texts_global)
    cat_texts_all.extend(face_texts_local)

    return cat_texts_all

def main_RN50(args , training_config):
    # prepare the model and text templates
    cat_texts_all = collect_separate_texts_v2(num_exif_scale=args.num_exif_scale)
    if args.text_enc == 'clip':
        if args.train_scratch:
            model, _, __, ___ = clip.load(args.clip_name, device=DEVICE, jit=False)
        else:
            model = CLIP_EXIF_Net(model_name="RN50", device=DEVICE,
                                  resume_img=args.resume, resume=args.resume_test,
                                  freezeText=args.freezeText,
                                  fixed_scale=args.fixed_scale, fixed_temp=args.fixed_temp).to(DEVICE)
    else:
        raise NotImplementedError

    model.cuda()
    texts_tokenize = clip.tokenize(cat_texts_all).to(DEVICE)

    # prepare the datasets
    ## training data, val data
    train_dataset = selectiveEXIF_data(train_mode=True, cls_face_mode=True, face_scale=1.3)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    # auto-lambda settings
    total_epoch = training_config["num_epochs"]
    device = DEVICE
    params = model.parameters()
    train_tasks = create_task_flags_sperate('all')
    pri_tasks = create_task_flags_sperate(args.task)
    train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-2]
    pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-2]
    logger.info('Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode '
                .format(train_tasks_str, pri_tasks_str))
    logger.info('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
                .format(args.weight.title(), args.grad_method.upper()))

    autol = BL_Optimizer(model, device, train_tasks, pri_tasks, args.weights_exif, args.weights_face)
    logger.info(f'meta_weights - {autol.meta_weights}')
    meta_weight_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)
    meta_optimizer = torch.optim.Adam([autol.meta_weights], lr=args.autol_lr)
    val_loader_autoL = train_dataloader

    # prepare the loss, optimizer, scheduler
    criterion_l2r = L2R_Loss()
    logger.info('categorical focal fidelity loss...')
    criterion_cls = categorical_focal_loss_fidelity()  # categorical_focal_loss_fidelity()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config["lr"],
        weight_decay=0.001)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.cosineTmax)

    # start training
    start_time = time.time()
    log_writer = SummaryWriter(log_dir=output_path)
    for epoch in range(training_config["num_epochs"]):
        logger.info('Auto-L: focal loss fidelity training...')

        train_loss, loss_ordinal, loss_cls, loss_face = train_one_epoch_FLfidAutoL_separate(train_dataloader, optimizer,
                                                                                            criterion_l2r,
                                                                                            criterion_cls,
                                                                                            model, texts_tokenize,
                                                                                            autol, val_loader_autoL,
                                                                                            meta_optimizer,
                                                                                            lr_scheduler,
                                                                                            epoch, training_config,
                                                                                            log_writer)

        model.eval()
        if args.save_model and ((epoch <= 23 and (epoch + 1) % args.test_model_freq == 0) or epoch == 0):
            if args.multi_gpu:
                save_model = model.module
            else:
                save_model = model
            checkpoint = {
                'epoch': epoch,
                'model': save_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict()
            }
            save_path = os.path.join(output_path, f'ckpt_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)

        lr_scheduler.step()

        meta_weight_ls[epoch] = autol.meta_weights.detach().cpu()

        logger.info(get_weight_str(meta_weight_ls[epoch], train_tasks))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch_FLfidAutoL_separate(data_loader, optimizer, criterion_l2r, criterion_cls, model, texts_tokenize,
                               autol, val_loader_autoL, meta_optimizer, scheduler,
                               epoch, training_config, log_writer):
    model.train()

    loss_meter = AverageMeter()
    loss_meter_val = AverageMeter()
    loss_exif_ordinal_meter = AverageMeter()
    loss_exif_categorical_meter = AverageMeter()
    loss_face_meter = AverageMeter()
    acc_meter_face = AverageMeter()
    acc_meter_make = AverageMeter()
    acc_meter_mm = AverageMeter()
    acc_meter_em = AverageMeter()
    acc_meter_wb = AverageMeter()
    acc_meter_ep = AverageMeter()
    batch_time = AverageMeter()
    num_steps = len(data_loader)

    start = time.time()
    end = time.time()
    val_dataset = iter(val_loader_autoL)
    for idx, batch in enumerate(data_loader):
        img, labels = batch
        img = img.to(DEVICE)

        # auto-lambda ------------------------------------------------
        val_samples, val_label = next(val_dataset)
        val_samples = val_samples.cuda(non_blocking=True)

        meta_optimizer.zero_grad()

        autol.unrolled_backward(img, labels, texts_tokenize,
                                val_samples, val_label,
                                scheduler.get_last_lr()[0],
                                scheduler.get_last_lr()[0], optimizer)

        meta_optimizer.step()
        # -----------------------------------------------------------

        optimizer.zero_grad()

        # forward
        logits_all_image, logits_all_text = model(img, texts_tokenize)

        ## for EXIF-ordinal tasks
        logits_all_EXIF_ordinal = logits_all_image[:, :12]
        iso_logits = logits_all_EXIF_ordinal[:, 0:3]
        av_logits = logits_all_EXIF_ordinal[:, 3:6]
        et_logits = logits_all_EXIF_ordinal[:, 6:9]
        fl_logits = logits_all_EXIF_ordinal[:, 9:12]
        ## EXIF: learning to rank
        loss_iso = criterion_l2r(iso_logits, labels['iso'].to(DEVICE))
        loss_av = criterion_l2r(av_logits, labels['av'].to(DEVICE))
        loss_et = criterion_l2r(et_logits, labels['et'].to(DEVICE))
        loss_fl = criterion_l2r(fl_logits, labels['fl'].to(DEVICE))
        loss_exif_ordinal = loss_iso + loss_av + loss_et + loss_fl

        ## for EXIF-categorical tasks
        logits_makes = logits_all_image[:, 12: 22]
        logits_mm = logits_all_image[:, 22: 30]
        logits_em = logits_all_image[:, 30: 37]
        logits_wb = logits_all_image[:, 37: 39]
        logits_ep = logits_all_image[:, 39: 45]
        ### multiclass classification
        loss_makes = criterion_cls(logits_makes, labels['makes'].to(DEVICE), num_classes=logits_makes.shape[1])
        loss_mm = criterion_cls(logits_mm, labels['mm'].to(DEVICE), num_classes=logits_mm.shape[1])
        loss_em = criterion_cls(logits_em, labels['em'].to(DEVICE), num_classes=logits_em.shape[1])
        loss_wb = criterion_cls(logits_wb, labels['wb'].to(DEVICE), num_classes=logits_wb.shape[1])
        loss_ep = criterion_cls(logits_ep, labels['ep'].to(DEVICE), num_classes=logits_ep.shape[1])

        loss_exif_categorical = loss_makes + loss_mm + loss_em + loss_wb + loss_ep

        loss_exif = loss_exif_ordinal + loss_exif_categorical

        # for face task: multilabel classification
        logits_all_face2text = logits_all_image[:, 45:]
        loss_face2text_coarse = Fidelity_Loss_multi()(logits_all_face2text[:, :2], labels['face2text'][:, 1].to(DEVICE),
                                                      num_classes=2)
        loss_face2text_fine = Fidelity_Loss_multi()(logits_all_face2text[:, 2:],
                                                    labels['face2text'][:, 2:].float().to(DEVICE),
                                                    num_classes=None)
        loss_face2text = loss_face2text_coarse + loss_face2text_fine
        loss_face = loss_face2text

        all_loss = [loss_iso, loss_av, loss_et, loss_fl,
                    loss_makes, loss_mm, loss_em, loss_wb, loss_ep,
                    loss_face2text_coarse, loss_face2text_fine]
        train_loss_tmp = [w * all_loss[i] for i, w in enumerate(autol.meta_weights)]
        total_loss = sum(train_loss_tmp)
        val_loss = sum(all_loss)

        total_loss.backward()
        optimizer.step()

        loss_meter.update(total_loss.item(), img.size(0))
        loss_meter_val.update(val_loss.item(), img.size(0))
        loss_exif_ordinal_meter.update(loss_exif_ordinal.item(), img.size(0))
        loss_exif_categorical_meter.update(loss_exif_categorical.item(), img.size(0))
        loss_face_meter.update(loss_face.item(), img.size(0))

        # face acc
        predicted_indices_l5 = predict_batch(logits_all_face2text[:, :2].softmax(1))
        acc_face = accuracy_batch(predicted_indices_l5, labels['face2text'][:, 1].to(DEVICE))
        acc_meter_face.update(acc_face, img.size(0))

        # exif categorical acc
        predicted_indices_makes = predict_batch(logits_makes.softmax(1))
        acc_makes = accuracy_batch(predicted_indices_makes, labels['makes'].to(DEVICE))
        acc_meter_make.update(acc_makes, img.size(0))

        predicted_indices_mm = predict_batch(logits_mm.softmax(1))
        acc_mm = accuracy_batch(predicted_indices_mm, labels['mm'].to(DEVICE))
        acc_meter_mm.update(acc_mm, img.size(0))

        predicted_indices_em = predict_batch(logits_em.softmax(1))
        acc_em = accuracy_batch(predicted_indices_em, labels['em'].to(DEVICE))
        acc_meter_em.update(acc_em, img.size(0))

        predicted_indices_wb = predict_batch(logits_wb.softmax(1))
        acc_wb = accuracy_batch(predicted_indices_wb, labels['wb'].to(DEVICE))
        acc_meter_wb.update(acc_wb, img.size(0))

        predicted_indices_ep = predict_batch(logits_ep.softmax(1))
        acc_ep = accuracy_batch(predicted_indices_ep, labels['ep'].to(DEVICE))
        acc_meter_ep.update(acc_ep, img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{training_config["num_epochs"]}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss {loss_meter_val.val:.4f} ({loss_meter_val.avg:.4f})\t'
                f'rank {loss_exif_ordinal_meter.val:.4f} ({loss_exif_ordinal_meter.avg:.4f})\t'
                f'cls {loss_exif_categorical_meter.val:.4f} ({loss_exif_categorical_meter.avg:.4f})\t'
                f'face {loss_face_meter.val:.4f} ({loss_face_meter.avg:.4f})\t'
                f'acc_makes {acc_meter_make.val:.4f} ({acc_meter_make.avg:.4f})\t'
                f'acc_mm {acc_meter_mm.val:.4f} ({acc_meter_mm.avg:.4f})\t'
                f'acc_em {acc_meter_em.val:.4f} ({acc_meter_em.avg:.4f})\t'
                f'acc_wb {acc_meter_wb.val:.4f} ({acc_meter_wb.avg:.4f})\t'
                f'acc_ep {acc_meter_ep.val:.4f} ({acc_meter_ep.avg:.4f})\t'
                f'acc_face {acc_meter_face.val:.4f} ({acc_meter_face.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

            log_writer.add_scalar('Train/total_loss', loss_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))  # *1000
            log_writer.add_scalar('Train/total_loss_val', loss_meter_val.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/exif_ordinal', loss_exif_ordinal_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/exif_categorical', loss_exif_categorical_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/face_loss', loss_face_meter.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))

            log_writer.add_scalar('Train/acc_makes', acc_meter_make.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/acc_mm', acc_meter_mm.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/acc_em', acc_meter_em.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/acc_wb', acc_meter_wb.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/acc_ep', acc_meter_ep.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))
            log_writer.add_scalar('Train/acc_face', acc_meter_face.val,
                                  int((idx / len(data_loader) + epoch) * len(data_loader)))

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, loss_exif_ordinal_meter.avg, loss_exif_categorical_meter.avg, loss_face_meter.avg


def setup_seed(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_path", default="checkpoints/wrapper.pth")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument('--multi_gpu', default=False, action='store_true', help='Bool type')
    parser.add_argument('--output', default='./output', type=str)
    parser.add_argument('--name', default='debug', type=str, help="name your experiment")
    parser.add_argument('--test_model_freq', type=int, default=5)
    parser.add_argument('--num_exif_scale', type=int, default=3)
    parser.add_argument('--cosineTmax', type=int, default=10)
    parser.add_argument('--save_model', action='store_true')


    parser.add_argument('--train_scratch', action='store_true')
    parser.add_argument('--clip_name', default='ViT-B/16', type=str, help='scratch based')


    parser.add_argument('--resume', default="/data0/mian3-2/literature_code/exif-as-language/wrapper_75_new.pth", type=str)
    parser.add_argument('--resume_test', type=str, default=None,
                        help='/data0/mian3-2/Experiments/exif-as-language-L2R-v2-debugExp/output/train/JE-OrdinalCateg/autol3e4_equalExFa-rn50exifinit_bz96_lr_1e5_FLfidelityV2-u-SM/ckpt_epoch_24.pth'
                        )
    parser.add_argument('--freezeText', action='store_true')
    parser.add_argument('--text_enc', type=str, default='clip')
    parser.add_argument('--fixed_scale', action='store_true')
    parser.add_argument('--fixed_temp', type=float, default=10)


    parser.add_argument('--autol_init', default=1.0, type=float, help='initialisation for auto-lambda')
    parser.add_argument('--autol_lr', default=1e-3, type=float, help='learning rate for auto-lambda')
    parser.add_argument('--task', default='exif', type=str, help='primary tasks, use all for MTL setting')
    parser.add_argument('--weight', default='autol', type=str, help='weighting methods: equal, dwa, uncert, autol')
    parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
    parser.add_argument('--weights_exif', default=1, type=float) # 就这俩有用
    parser.add_argument('--weights_face', default=0.1, type=float)


    args = parser.parse_args()

    setup_seed(42)

    output_path = os.path.join(args.output, args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = create_logger(output_dir=output_path, name=f"{args.name}")
    logger.info(' '.join(list(sys.argv)))

    training_config = {
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "path": args.save_model_path
    }

    main_RN50(args, training_config)