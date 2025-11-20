import copy
import random
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverprox import FedProx
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.trainmodel.mnist_model import MnistNet
from FLAlgorithms.trainmodel.fashionmnist_model import FMnistNet
from FLAlgorithms.trainmodel.cifar_model import ResNet18_cifar, ResNet34, ResNet50
from FLAlgorithms.trainmodel.cifar100_model import ResNet18cifar100
from utils.plot_utils import *
import torch
import datetime
from torchvision import datasets, transforms

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True  # cudnn
random.seed(1)
np.random.seed(1)

# central_dataset is a idx list
def central_dataset_iid(dataset, dataset_size):
    all_idxs = [i for i in range(len(dataset))]
    central_dataset = set(np.random.choice(
        all_idxs, dataset_size, replace=False))
    return central_dataset

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, malnum, poisonratio, attack_method,
         per_epoch, attack_start, oneshot, clip_rate, defense, server_dataset, tau, local_bs, plr_class, server_lr, wrong_mal, right_ben, turn, noise):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    trigger_patten = []
    trigger_list = []

    # load trigger
    if dataset == 'Mnist' or dataset == 'FashionMnist' or dataset=='Cifar10':
        for i in range(10, 20, 1):
            for j in range(10, 20, 1):
                trigger_patten.append([i, j])

        malclient = ['f_00007', 'f_00001', 'f_00062', 'f_00020', 'f_00096', 'f_00085', 'f_00051', 'f_00043',
                     'f_00037', 'f_00058']
        # malclient = ["kkkkkkkkkk"]
        poison_label = 0
        if dataset == 'Mnist' or dataset == 'FashionMnist':
            intinal_trigger = torch.zeros((1, 28, 28)).float().to(device)
        elif dataset == 'Cifar10':
            intinal_trigger = torch.zeros((3, 32, 32)).float().to(device)

        for i in trigger_patten:
            intinal_trigger[0][i[0]][i[1]] = 0.5

        for i in range(10):
            trigger_list.append(copy.deepcopy(intinal_trigger))

    else:
        raise ValueError("dataset name wrong!")

    print(malclient)

    for i in range(times):  # 重复实验
        print("---------------Running time:------------", i)
        # Generate model

        if dataset == 'Mnist':
            model = MnistNet(name="global", created_time=current_time).to(device)
            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(
                '../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST(
                '../data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            central_dataset_full = datasets.EMNIST(
                '../data/emnist', split='balanced', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            )
        elif dataset == 'FashionMnist':
            model = FMnistNet(name="global", created_time=current_time).to(device)
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
            dataset_test = datasets.FashionMNIST(
                '../data/', train=False, download=True, transform=trans_mnist)
            central_dataset_full = datasets.KMNIST(
                '../data/kmnist', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
            )

        elif dataset == 'Cifar10':
            model = ResNet18_cifar(name="global", created_time=current_time).to(device)
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_test = datasets.CIFAR10(
                '../data/cifar', train=False, download=True, transform=trans_cifar)
            central_dataset_full = datasets.CIFAR100(
                '../data/cifar100', train=False, download=True, transform=trans_cifar
            )

        else:
            raise ValueError("dataset name wrong!")

        # for some defenses
        print("Now we use another dataset for validation")
        central_dataset = central_dataset_iid(central_dataset_full, server_dataset)

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                            malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                            attack_method=attack_method, per_epoch=per_epoch, defense=defense,
                            central_dataset=central_dataset, dataset_test=dataset_test, tau=tau, local_bs=local_bs, plr_class=plr_class,
                            server_lr=server_lr, wrong_mal=wrong_mal, right_ben=right_ben, turn=turn, noise=noise)

        elif algorithm == "FedProx":
            server = FedProx(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                             local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                             malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                             attack_method=attack_method, per_epoch=per_epoch, defense=defense,
                             central_dataset=central_dataset, dataset_test=dataset_test, tau=tau, local_bs=local_bs,
                             plr_class=plr_class,
                             server_lr=server_lr
                             )
        elif algorithm == "pFedMe":
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                            local_epochs, optimizer, numusers, i, False, current_time=current_time, malnum=malnum,
                            malclient=malclient, poisonratio=poisonratio, poison_label=poison_label,
                            attack_method=attack_method, per_epoch=per_epoch, defense=defense,
                            central_dataset=central_dataset, dataset_test=dataset_test, tau=tau, local_bs=local_bs, plr_class=plr_class,
                            server_lr=server_lr, wrong_mal=wrong_mal, right_ben=right_ben, turn=turn, noise=noise, K=K, personal_learning_rate=personal_learning_rate)


        else:
            raise ValueError("alg name wrong!")

        final_trigger_list = server.train(pattern=trigger_patten, trigger=trigger_list, per_epoch=per_epoch,
                                          attack_start=attack_start, oneshot=oneshot, clip_rate=clip_rate,
                                          defense=defense)

        print(final_trigger_list[0])

        # local finetuning
        server.send_parameters()  # 将当前的全局模型分给每一个用户 deepcopy

        # Evaluate the final global model
        print("Evaluate the final global model")
        globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_acc, global_test_mean_mal_acc = server.evaluate()  # 分发了模型，当前用户的模型是全局模型， 输出所有本地数据在全局模型上测试的准确率
        globaltestasr, globaltrainasr, globaltrainasrloss, global_test_mean_benign_asr, global_test_mean_mal_asr = server.poison_evaluate(
            trigger=final_trigger_list, pattern=trigger_patten)  # 输出所有本地数据在全局模型上测试的准确率

        print("")

        # Evaluate gloal model on user for each interation
        print("Evaluate the final global model with a few step update, which is personalized model")
        pertestacc, pertrainacc, pertrainloss, poiperasr, poipertrainasr, poiperloss, per_mean_ben_asr, per_mean_mal_asr, per_mean_ben_acc, per_mean_mal_acc = server.evaluate_one_step(
            per_epoch, trigger=final_trigger_list, pattern=trigger_patten)  # 本地finetune一次，在本地模型上再测试   --个性化模型准确率

        # # clean train
        # server.trainClean()
        # server.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10",
                        choices=["Mnist", "FashionMnist", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn", "VGG16", "resnet", "lenet"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedMe",
                        choices=["pFedMe", "PerAvg-FO", "PerAvg-HF", "FedAvg", "FedProx", "Ditto", "SCAFFOLD", "FedBN"])
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--malclient", type=int, default=10, help="number of malicious client")
    parser.add_argument("--attack_start", type=int, default=10, help="the start attack iteration")
    parser.add_argument("--mal_local_epoch", type=int, default=20)
    parser.add_argument("--poisoning_per_batch", type=int, default=5, help="the poison ratio")
    parser.add_argument("--attack_method", type=str, default='attackall',
                        choices=['attackall'])
    parser.add_argument("--attack_goal", type=str, default='attackall', choices=['attackone', 'attackall'])
    parser.add_argument("--per_epoch", type=int, default='1', help='the epoch for local finetune')
    parser.add_argument("--descrip", type=str, help="the gradient mask ratio")
    parser.add_argument("--oneshot", type=int, default=0, help="one shot attack", choices=[1, 0])
    parser.add_argument("--clip_rate", type=int, default=0, help="one shot attack scale")
    parser.add_argument("--defense", type=str, default='none', help="defense method",
                        choices=['none', 'mkrum', 'trim', 'geminiguard', 'fltrust', 'flare', 'flshield','flame','earlyshield'])

    #parameters for geminiguard
    parser.add_argument('--server_dataset', type=int, default=100, help="number of dataset in server")
    parser.add_argument('--tau', type=float, default=0.8, help="threshold of LPA_ER")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--plr_class', type=int, default=6, help="get PLRs of a specific class")

    #parameters for fltrust
    parser.add_argument('--server_lr', type=float, default=1, help="number of dataset in server using in fltrust")

    # parameters for flame
    parser.add_argument('--wrong_mal', type=int, default=0)
    parser.add_argument('--right_ben', type=int, default=0)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--noise', type=float, default=0.001)

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Attack method:{}".format(args.attack_method))
    print("Defense method:{}".format(args.defense))
    print("Attack goal:{}".format(args.attack_goal))
    print("Start attack iteration:{}".format(args.attack_start))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Per_local epoch:{}".format(args.per_epoch))
    print("one shot:{}".format(args.oneshot))
    print("scale rate:{}".format(args.clip_rate))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        lamda=args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times=args.times,
        malnum=args.malclient,
        poisonratio=args.poisoning_per_batch,
        attack_method=args.attack_method,
        per_epoch=args.per_epoch,
        attack_start=args.attack_start,
        oneshot=args.oneshot,
        clip_rate=args.clip_rate,
        defense=args.defense,

        server_dataset=args.server_dataset,
        tau=args.tau,
        local_bs=args.local_bs,
        plr_class=args.plr_class,

        server_lr=args.server_lr,

        wrong_mal = args.wrong_mal,
        right_ben = args.right_ben,
        turn = args.turn,
        noise = args.noise
    )
