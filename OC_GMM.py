import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pickle, time, os, argparse, random

from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset.OC_dataset import GatherDataset
from utils import save_features, tsne_plot_save_dir
from model import CLIP_EXIF_Net

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ------------- 1. extract features ---------------
def extract_training_features(model, real_tag='celeba', save_dir='', real_num=25000):
    dataset = GatherDataset(
        real_item_tag=real_tag, only_real=True, training=True,
        random_subset=True,
        random_subset_num_real=real_num,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4
    )
    print(f'length of photographic dataset: {len(dataset)}')

    fea_all = []
    for batch in tqdm(dataloader):
        img = batch['img'].to(DEVICE)

        with torch.no_grad():
            features = model.encode_image(img)
            features = features / features.norm(dim=1, keepdim=True)
        fea_all.append(features)
    fea_all = torch.cat(fea_all, dim=0)
    save_name = str(len(dataset))+'_'+real_tag
    save_dir = os.path.join(save_dir, real_tag)
    os.makedirs(save_dir, exist_ok=True)
    save_features(fea_all, save_name, save_dir)



def extract_testing_features(model, real_tag='celeba', save_dir='', plot_tsne=False):
    os.makedirs(save_dir, exist_ok=True)
    fake_items = [
        'stylegan2','vqgan',
        'ldm', 'ddim', 'sdv21',
        'freeDom', 'hps', 'midj', 'sdxl'
    ]

    for dataset_item in fake_items:
        dataset_real = GatherDataset(
            real_item_tag=real_tag, only_real=True, only_fake=False,
            fake_items_tag=dataset_item,
            training=False,
            random_subset=True,
            random_subset_num_fake=5000,
            random_subset_num_real=5000,
        )
        print(f'length of {dataset_item} - photographic: {len(dataset_real)}')
        dataloader_real = DataLoader(
            dataset_real,
            batch_size=256,
            shuffle=True,
            num_workers=4
        )


        fea_all = []
        label_all_tsne = []
        fea_all_tsne = []
        for batch in tqdm(dataloader_real):
            img = batch['img'].to(DEVICE)
            label = [1] * img.size(0)

            with torch.no_grad():
                features = model.encode_image(img)
                features = features / features.norm(dim=1, keepdim=True)
            fea_all.append(features)

            fea_all_tsne.append(features)
            label_all_tsne.extend(label)

        fea_all = torch.cat(fea_all, dim=0)
        save_name = dataset_item+'_'+real_tag + '_real'
        save_features(fea_all, save_name, save_dir)
        print(f'real {real_tag} features saved...')

        dataset_fake = GatherDataset(
            real_item_tag=real_tag, only_fake=True, only_real=False,
            fake_items_tag=dataset_item, training=False,
            random_subset=True,
            random_subset_num_fake=5000,
            random_subset_num_real=5000,
        )
        dataloader_fake = DataLoader(
            dataset_fake,
            batch_size=256,
            shuffle=True,
            num_workers=4
        )
        print(f'length of {dataset_item} - generated: {len(dataset_fake)}')

        fea_all = []
        for batch in tqdm(dataloader_fake):
            img = batch['img'].to(DEVICE)
            label = [0] * img.size(0)

            with torch.no_grad():
                features = model.encode_image(img)
                features = features / features.norm(dim=1, keepdim=True)
            fea_all.append(features)

            fea_all_tsne.append(features)
            label_all_tsne.extend(label)

        fea_all = torch.cat(fea_all, dim=0)
        save_name = dataset_item + '_fake'
        save_features(fea_all, save_name, save_dir)

        print(f'fake {dataset_item} features saved...')

        if plot_tsne:
            fea_all_tsne = torch.cat(fea_all_tsne, dim=0)
            result_dir = os.path.join(save_dir, 'tsne')
            tsne_plot_save_dir(fea_all_tsne.detach().cpu().numpy(), label_all_tsne, result_dir,
                               savename='binary_tsne_'+dataset_item+'.png')


# ------------- 2. train GMM and test ---------------
def train_GMM_sklearn(feature_path="",
                      path_dir="",
                      n_comp=12):
    gmm = GaussianMixture(n_components=n_comp, init_params='k-means++', random_state=2024)

    # load the feature data
    features = torch.load(feature_path)
    print(features.shape)

    start_time = time.time()
    try:
        gmm.fit(features.detach().cpu().numpy())
    except:
        gmm.fit(features.numpy())
    end_time = time.time()
    runtime = end_time - start_time

    # 转换为小时、分钟和秒
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = int(runtime % 60)
    # 打印运行时间
    print("time：{}hr {}min {}s".format(hours, minutes, seconds))

    os.makedirs(path_dir, exist_ok=True)
    with open(path_dir + 'gmm.pkl', 'wb') as file:
        pickle.dump(gmm, file)


def test_GMM_sklearn(gmm_model_path='', real_tag='',
                     feature_dir_path='',
                     pred_save_path=''):
    with open(gmm_model_path, 'rb') as file:
        gmm = pickle.load(file)

    fake_items = [
        'stylegan2','vqgan',
        'ldm', 'ddim', 'sdv21',
        'freeDom', 'hps', 'midj', 'sdxl']

    for dataset_item in fake_items:
        feature_path_real = feature_dir_path + dataset_item + '_'+real_tag + '_real.pth'
        features_real = torch.load(feature_path_real)
        feature_path_fake = feature_dir_path + dataset_item + '_fake.pth'
        features_fake = torch.load(feature_path_fake)

        print(f'length of {dataset_item} - photographic: {len(features_real)} - generated: {len(features_fake)}')

        real_logp = []
        log_likelihoods_real = gmm.score_samples(features_real.cpu())
        real_logp.extend(log_likelihoods_real.tolist())

        fake_logp = []
        log_likelihoods = gmm.score_samples(features_fake.cpu())
        fake_logp.extend(log_likelihoods.tolist())

        log_likelihood = {}
        log_likelihood['real'] = real_logp
        log_likelihood[dataset_item] = fake_logp

        # save likelihood
        savename = 'likelihood_' + dataset_item
        os.makedirs(pred_save_path, exist_ok=True)
        filename = pred_save_path + savename + '.pickle'
        with open(filename, 'wb') as file:
            pickle.dump(log_likelihood, file)

        print(f"Data saved to file: {filename}")


def GMM_compute_thr(real_train_features_path="", gmm_path="",
                    false_alarm=0.05):
    with open(gmm_path, 'rb') as file:
        gmm = pickle.load(file)

    real_logp = []
    features_real = torch.load(real_train_features_path)
    log_likelihoods_real = gmm.score_samples(features_real.cpu())

    real_logp.extend(log_likelihoods_real.tolist())
    real_logp = sorted(real_logp)
    threshold_index = int(len(real_logp) * false_alarm)
    threshold = real_logp[threshold_index]

    print('threshold:{}'.format(threshold))
    return threshold


def GMM_compute_mAP_auc(thr, pred_save_path=''):
    fake_items = [
        'stylegan2', 'vqgan',
        'ldm', 'ddim', 'sdv21',
        'freeDom', 'hps', 'midj', 'sdxl']

    mAUC_meter = []
    mAP_meter = []
    mAcc_meter = []
    for dataset_item in fake_items:
        logp = []
        label = []
        logp_dict_path = pred_save_path +"likelihood_"+dataset_item+".pickle"
        with open(logp_dict_path, 'rb') as file:
            likelihood_dict = pickle.load(file)
        real_logp = likelihood_dict['real']
        fake_logp = likelihood_dict[dataset_item]

        real_label = [1] * len(real_logp)
        fake_label = [0] * len(fake_logp)
        print(f'photographic: {len(real_logp)} - generated: {len(fake_logp)}')
        logp.extend(real_logp)
        logp.extend(fake_logp)
        label.extend(real_label)
        label.extend(fake_label)

        binary_predictions = [1 if pred >= thr else 0 for pred in logp]
        correct_predictions = sum([1 for pred, label in zip(binary_predictions, label) if pred == label])
        accuracy = correct_predictions / len(label) * 100
        print('Acc:{}'.format(accuracy))

        auc = roc_auc_score(label, logp) * 100
        print(f"{dataset_item} AUC: {auc:.2f}")

        map_score = average_precision_score(label, logp) * 100
        print(f"{dataset_item} AP: {map_score:.2f}")

        mAcc_meter.append(accuracy)
        mAUC_meter.append(auc)
        mAP_meter.append(map_score)

    # calculate mean number
    print(f"Mean number of mAcc: {np.mean(mAcc_meter): .4f}")
    print(f"Mean number of mAUC: {np.mean(mAUC_meter): .4f}")
    print(f"Mean number of mAP: {np.mean(mAP_meter): .4f}")


if __name__ == '__main__':
    seed = 2024
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        default='./pretrained/OC.pth',
                        type=str)
    parser.add_argument('--save_dir', type=str,
                        default='./output/test/')
    parser.add_argument('--output_name', type=str,
                        default='OC_GMM')
    parser.add_argument('--n_comp_GMM', type=int, default=10)

    args = parser.parse_args()


    model = CLIP_EXIF_Net(model_name='RN50', device=DEVICE, resume=args.resume).to(DEVICE)
    model.eval()


    extract_training_features(model, real_tag='celeba',
                              save_dir=args.save_dir + args.output_name + "/train")

    extract_testing_features(model, real_tag='celeba',
                             save_dir=args.save_dir + args.output_name + "/test",
                             plot_tsne=False)

    train_GMM_sklearn(  # check number of GMM components
        feature_path=args.save_dir + args.output_name + "/train/celeba/25000_celeba.pth",
        path_dir=args.save_dir + args.output_name + "/train/", n_comp=args.n_comp_GMM)

    test_GMM_sklearn(
        gmm_model_path=args.save_dir + args.output_name + "/train/gmm.pkl",
        real_tag='celeba',
        feature_dir_path=args.save_dir + args.output_name + "/test/",
        pred_save_path=args.save_dir + args.output_name + "/gmm_preds/")

    thr = GMM_compute_thr(real_train_features_path=args.save_dir + args.output_name + "/train/celeba/25000_celeba.pth",
                          gmm_path=args.save_dir + args.output_name + "/train/gmm.pkl",
                          false_alarm=0.05)

    GMM_compute_mAP_auc(
        thr,
        pred_save_path=args.save_dir + args.output_name + "/gmm_preds/")
