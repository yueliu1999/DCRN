from train import *
from DCRN import DCRN


if __name__ == '__main__':
    # setup
    setup()

    # data pre-precessing: X, y, A, A_norm, Ad
    X, y, A = load_graph_data(opt.args.name, show_details=False)
    A_norm = normalize_adj(A, self_loop=True, symmetry=False)
    Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)

    # to torch tensor
    X = numpy_to_torch(X).to(opt.args.device)
    A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
    Ad = numpy_to_torch(Ad).to(opt.args.device)

    # Dual Correlation Reduction Network
    model = DCRN(n_node=X.shape[0]).to(opt.args.device)

    # deep graph clustering
    acc, nmi, ari, f1 = train(model, X, y, A, A_norm, Ad)
    print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1))
