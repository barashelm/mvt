import os
import random
import thop
import torch

from utils import logger
from utils.logger import line_seg

from models import csinetp, transnet, stnet, mnet, crnet, crissnet, mvt
__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        # torch.cuda.set_device(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("=> Running on GPU %d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(model_name, args):
    # Model loading
    if model_name == 'csinetp':
        model = csinetp(reduction=args.cr)
    elif model_name == 'crnet':
        model = crnet(reduction=args.cr)
    elif model_name == 'transnet':
        model = transnet(reduction=args.cr)
    elif model_name == 'stnet':
        model = stnet(reduction=args.cr)
    elif model_name == 'mnet':
        model = mnet(reduction=args.cr)
    elif model_name == 'crissnet':
        model = crissnet(reduction=args.cr)
    elif model_name == 'mvt':
        model = mvt(reduction=args.cr)


    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    macs, params = thop.profile(model, inputs=(image,), verbose=False)
    macs, params = thop.clever_format([macs, params], "%.5e")

    # Model info logging
    logger.info(f'=> Model Name: {model_name} [pretrained: {args.pretrained}]')
    logger.info(f'=> Scenario = {args.scenario} | Compression Ratio: {args.cr}')
    logger.info(f'=> Params:{params} MACs:{macs}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
