import time
import torch
from tqdm import tqdm

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                clusterecost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)

    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    """
    =============================================================================
    PHASE 2: LAST LAYER ONLY TRAINING
    =============================================================================
    Purpose: Freeze all components except the final classification layer
    Trainable: ONLY last layer (classification head)
    Frozen: Feature backbone, add-on layers, prototype vectors
    Use case: Fine-tuning after prototype pushing
    =============================================================================
    """
    if hasattr(model, 'module'):
        model = model.module
    
    # Freeze feature extractor backbone (ResNet101, VGG, etc.)
    for p in model.features.parameters():
        p.requires_grad = False
    
    # Freeze add-on layers (feature processing layers)
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    
    # Freeze prototype vectors (learned prototypes)
    model.prototype_vectors.requires_grad = False
    
    # ONLY train the last layer (classification head)
    # This layer maps prototype activations to class predictions
    for p in model.last_layer.parameters():
        p.requires_grad = True


def warm_only(model, log=print):
    """
    =============================================================================
    PHASE 0: WARMUP TRAINING
    =============================================================================
    Purpose: Initialize prototypes and add-on layers while keeping backbone frozen
    Trainable: Add-on layers, ASPP layers, prototype vectors, last layer
    Frozen: Feature backbone (ResNet101, VGG, etc.)
    Use case: Initial training to learn meaningful prototypes
    =============================================================================
    """
    # Get ASPP (Atrous Spatial Pyramid Pooling) parameters from DeepLab
    # These are the dilated convolution layers that capture multi-scale features
    aspp_params = [
        model.features.base.aspp.c0.weight,  # ASPP conv 1x1
        model.features.base.aspp.c0.bias,
        model.features.base.aspp.c1.weight,  # ASPP conv 3x3, rate=6
        model.features.base.aspp.c1.bias,
        model.features.base.aspp.c2.weight,  # ASPP conv 3x3, rate=12
        model.features.base.aspp.c2.bias,
        model.features.base.aspp.c3.weight,  # ASPP conv 3x3, rate=18
        model.features.base.aspp.c3.bias
    ]

    if hasattr(model, 'module'):
        model = model.module
    
    # Freeze feature extractor backbone (ResNet101, VGG, etc.)
    # Keep pretrained features intact during warmup
    for p in model.features.parameters():
        p.requires_grad = False
    
    # Train add-on layers (feature processing layers)
    # These layers process features from the frozen backbone
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    
    # Train prototype vectors (learnable prototypes)
    # These represent characteristic features of each class
    model.prototype_vectors.requires_grad = True
    
    # Train last layer (classification head)
    # Maps prototype activations to class predictions
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    # Train ASPP layers (multi-scale feature extraction)
    # These are important for capturing features at different scales
    for p in aspp_params:
        p.requires_grad = True


def joint(model, log=print):
    """
    =============================================================================
    PHASE 1: JOINT TRAINING
    =============================================================================
    Purpose: Fine-tune entire network with different learning rates for components
    Trainable: ALL components (backbone, add-on layers, prototypes, last layer)
    Strategy: Lower LR for backbone, higher LR for new components
    Use case: Main training phase after warmup initialization
    =============================================================================
    """
    if hasattr(model, 'module'):
        model = model.module
    
    # Train feature extractor backbone (ResNet101, VGG, etc.)
    # Use lower learning rate to preserve pretrained features
    for p in model.features.parameters():
        p.requires_grad = True
    
    # Train add-on layers (feature processing layers)
    # Use higher learning rate for new components
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    
    # Train prototype vectors (learnable prototypes)
    # These are the core components that represent class features
    model.prototype_vectors.requires_grad = True
    
    # Train last layer (classification head)
    # Maps prototype activations to class predictions
    for p in model.last_layer.parameters():
        p.requires_grad = True
