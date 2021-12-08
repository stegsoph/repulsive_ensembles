import matplotlib
import torch
from utils.distributions import Unorm_post
from methods.method_utils import create_method
matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi']= 150
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import ood_metrics_entropy
import uncertainty_metrics as um
import seaborn as sns
import torchvision 
sns.set()

def train(data_train,data_test, ensemble, device, config,writer):
    """Train the particles using a specific ensemble.

    Args:
        data: A DATASET instance.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        writer: The tensorboard summary writer.
    """

    # particles.train()

    #W = ensemble.particles.clone()

    W = ensemble.particles
    samples = []
    
    if config.optim == 'Adam':
        optimizer = torch.optim.Adam([W], config.lr, weight_decay=config.weight_decay,
                                     betas=[config.adam_beta1, 0.999])
    else:
        optimizer = torch.optim.SGD([W], config.lr)
        
    prior = torch.distributions.normal.Normal(torch.zeros(ensemble.net.num_params).to(device),
                          torch.ones(ensemble.net.num_params).to(device) * config.prior_variance)
    
    if config.method == 'f_s_SVGD': 
        add_prior = False 
    else:
        add_prior = True
    
    noise = torch.distributions.normal.Normal(torch.zeros(data_train.in_shape[0]**2).to(device),
                          torch.ones(data_train.in_shape[0]**2).to(device))

    P = Unorm_post(ensemble, prior, config, data_train.num_train_samples,add_prior)

    log_scale = torch.log2(torch.tensor(data_train.out_shape[0],dtype=torch.float))
    
    #--------------------------------------------------------------------------------
    # SVGD ALGORITHM SPECIFICATIONS
    #--------------------------------------------------------------------------------
    
    method = create_method(config,P,optimizer, device = device )

    #--------------------------------------------------------------------------------
    # CREATING TESTING AND CORRUPTED DATASET FOR OOD 
    #--------------------------------------------------------------------------------
    
    x_test_0 = torch.tensor(data_train.get_test_inputs(), dtype=torch.float).to(device)
    y_test_0 = torch.tensor(data_train.get_test_outputs(), dtype=torch.float).to(device)
    test_labels = torch.argmax(y_test_0, 1).type(torch.int).to(device)
    
    #intensity = [0,0.5,1,1.5,2.0]
    #intensity =  np.linspace(0.05,0.1,5)
    intensity = [0.1,0.5,1,1.5,2.0,5.0]
    kernels = [torchvision.transforms.GaussianBlur(7, sigma=inte) for inte in intensity]
    
    if config.dataset == 'cifar':
        s = (-1,3,32,32)
    else: 
        s = (-1,1,28,28)

    #The gaussian blur like this is actually convolving different pictures, the effect is good for us because it is mixing 
    #the categories but maybe it is not what we would like to have 
    #blurred = [torch.tensor(gaussian(x_test_0,sigma=inte,multichannel=False), dtype=torch.float) for inte in intensity]
    
    blurred_inj = [x_test_0 + torch.empty(x_test_0.shape).normal_(mean=0,std=inte).to(device) for inte in intensity]
    test_shape = x_test_0.shape
    blurred = [ke.forward(x_test_0.reshape(s)).reshape(test_shape) for ke in kernels]
    #fig=plt.figure(figsize=(15, 15))

    #for i,inte in enumerate(intensity):
    #    fig.add_subplot(1, len(intensity), i+1)
    #    test_b = gaussian(x_test_0[5],sigma=inte,multichannel=False)
    #    plt.imshow(test_b.reshape((28,28)))

    
    #writer.add_figure('Data Corruption', plt.gcf(),0, close=not config.show_plots)
    #plt.close()
    
    #Use this if ood are given by the other dataset
    x_test_ood = torch.tensor(data_test.get_test_inputs(), dtype=torch.float).to(device)[:10000]
    y_test_ood = torch.tensor(data_test.get_test_outputs(), dtype=torch.float).to(device)[:10000]
    test_labels_ood = torch.argmax(y_test_ood, 1).type(torch.int).to(device)

    #mem = torch.cuda.memory_allocated(device)
    #print('Pre-train:'+' Mem:'+"{:.3f} GB".format(mem / 1024 ** 3), flush = True)
    
        
    print('-------------------Start training------------------')
    #--------------------------------------------------------------------------------
    # SVGD TRAINING
    #--------------------------------------------------------------------------------

    for i in range(config.epochs):

        optimizer.zero_grad()

        batch_train = data_train.next_train_batch(config.batch_size)
        batch_test = data_train.next_test_batch(config.batch_size)
        batch_ood = data_test.next_train_batch(config.batch_size)
        X = data_train.input_to_torch_tensor(batch_train[0], device, mode='train')
        T = data_train.output_to_torch_tensor(batch_train[1], device, mode='train')
        X_t = data_train.input_to_torch_tensor(batch_test[0], device, mode='train')
        T_t = data_train.output_to_torch_tensor(batch_test[1], device, mode='train')
        

        #X_ood = data_train.input_to_torch_tensor(batch_ood[0], device, mode='train')
        #T_ood = data_train.output_to_torch_tensor(batch_ood[1], device, mode='train')
        
        #Adding noise to test as oood 
        #x_test_ood = x_test_0 + noise.sample(torch.Size([x_test_0.shape[0]]))
        
        

        # if config.clip_grad_value != -1:
        #    torch.nn.utils.clip_grad_value_(optimizer.param_groups[0]['params'],
        #                                    config.clip_grad_value)
        # elif config.clip_grad_norm != -1:
        #torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],config.clip_grad_norm)
        #torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'],2,2)

        
        if config.method == 'SVGD_annealing':
            driving,repulsive = method.step(W, X, T,i)
        elif config.method == 'SVGD_debug':
            driving,repulsive = method.step(W, X, T)
        elif config.method == 'SGD' or config.method == 'SGLD':
            method.step(W, X, T)
        elif config.method == 'f_p_SVGD' or config.method == 'mixed_f_p_SVGD' or config.method == 'f_s_SVGD'  or config.method == 'f_SGD':
            #noise_samples = noise.sample(torch.Size([config.batch_size]))
            if config.where_repulsive == 'train':
                driving,repulsive = method.step(W, X, T,i, None)
            elif config.where_repulsive == 'noise':
                blurred_train = kernels[2].forward(X.reshape(s)).reshape(X.shape) #the number kernels decide intensity
                driving,repulsive = method.step(W,X,T,i,blurred_train)
            elif config.where_repulsive == 'test':
                driving,repulsive = method.step(W,X,T,i,X_t)
            elif config.where_repulsive == 'inj':
                inj_train = X + torch.empty(X.shape).normal_(mean=0,std=0.075).to(device)
                driving,repulsive = method.step(W,X,T,i,inj_train)


            #driving,repulsive = method.step(W, X, T,i, None)
        elif config.method == 'SVGLD':
            driving,repulsive,langevin_noise = method.step(W, X, T,i)

        else:
            driving,repulsive = method.step(W, X, T,i)

        if i % 10000 == 0 or i==config.epochs-1:
            train_loss, train_pred = P.log_prob(W, X, T, return_loss=True)
            test_loss, test_pred = P.log_prob(W, x_test_0, y_test_0, return_loss=True)
            writer.add_scalar('train/train_loss', train_loss, i)
            writer.add_scalar('test/test_loss', test_loss, i)
            if 'driving' in locals():
                writer.add_scalar('train/driving_force', torch.mean(driving.abs()), i)
                writer.add_scalar('train/repulsive_force', torch.mean(repulsive.abs()), i)
                writer.add_scalar('train/forces_ratio', torch.mean(repulsive.abs())/torch.mean(driving.abs()), i)
            if config.method == 'SVGLD': 
                writer.add_scalar('train/langevin_noise', torch.mean(langevin_noise.abs()), i)
                #writer.add_scalar('train/bandwith', K.h, i)
            if hasattr(method, 'ann_schedule'):
                writer.add_scalar('train/annealing', method.ann_schedule[i], i)
            if ensemble.net.classification:
                print(test_pred.shape)
                print(train_pred.shape)
                std_prob_test = test_pred.std(0).mean(1)
                Y = torch.mean(train_pred, 0)
                Y_t = torch.mean(test_pred, 0)
                entropies_test = -torch.sum(torch.log2(Y_t + 1e-20)/log_scale * Y_t, 1)
                entropies_train = -torch.sum(torch.log2(Y + 1e-20)/log_scale * Y, 1)


                train_accuracy = (torch.argmax(Y, 1) == torch.argmax(T, 1)).sum().item() / Y.shape[0] * 100
                test_accuracy = (torch.argmax(Y_t, 1) == test_labels).sum().item() / Y_t.shape[0] * 100
                
                nll = -torch.log((y_test_0.expand_as(Y_t) *(Y_t)).max(1)[0].mean(0))
                print(nll)
                writer.add_scalar('train/accuracy', train_accuracy, i)
                writer.add_scalar('test/accuracy', test_accuracy, i)
                writer.add_scalar('test/entropy', entropies_test.mean(), i)
                writer.add_scalar('train/entropy', entropies_train.mean(), i)
                writer.add_scalar('test/std', std_prob_test.mean(), i)
                writer.add_scalar('test/nll', nll, i)




                print('Train iter:',i, ' train acc:', train_accuracy, 'test_acc', test_accuracy, flush = True)
                
                

        if (i % 10000 == 0 and i!= 0) or i==config.epochs-1:
            #--------------------------------------------------------------------------------
            # accurcacy vs uncertainty plots
            #--------------------------------------------------------------------------------
            _, indices = torch.sort(entropies_test, dim = 0, )
            acc = []
            samples = np.arange(1,x_test_0.shape[0],100)
            correct_sorted = (torch.argmax(Y_t, 1) == test_labels)[indices.squeeze()]
            for n_points in samples:
                acc.append(correct_sorted[:n_points].sum().item() / Y_t[:n_points].shape[0])

            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.plot(samples, acc, markersize=8, c='b', marker="o", label='accuracy')
            ax1.set_xlabel('Number of points included')    
            ax1.set_ylabel('Accuracy')    
            ax1.legend()

            writer.add_figure('accuracy_uncertainty', plt.gcf(),i, close=not config.show_plots)
            plt.close()
            
            #--------------------------------------------------------------------------------
            # OOD TESTS
            #--------------------------------------------------------------------------------
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

            softmax_ood = ensemble.forward(x_test_ood)[0]
            
            std_prob_ood = softmax_ood.std(0).mean(1)

            pred_ood = torch.argmax(softmax_ood,2)

            average_prob_ood = softmax_ood.mean(0)
            ood_confidence = torch.max(average_prob_ood,1)[0]
            test_confidence = torch.max(Y_t,1)[0]

            #KL_uniform = -torch.sum(average_prob * torch.log2(
            #    (torch.ones(average_prob.shape[1]) / average_prob.shape[1] )/ average_prob + 1e-20)/log_scale, 1)
            #KL_uniform[KL_uniform != KL_uniform] = 0
            #writer.add_scalar('ood_metrics/AV_KL_uniform', KL_uniform.mean(), i)
            entropies_ood = -torch.sum(torch.log2(average_prob_ood + 1e-20)/log_scale * average_prob_ood, 1)
            writer.add_scalar('ood_metrics/Av_entropy/', entropies_ood.mean(), i)
            writer.add_scalar('ood_metrics/entropy_ratio/', entropies_test.mean()/entropies_ood.mean(), i)
            writer.add_scalar('ood_metrics/std_ood/', std_prob_ood.mean(), i)




            # diversity = torch.mean(torch.min(pred_ood.sum(0), torch.tensor(pred_ood.shape[0])-pred_ood.sum(0)          )/(pred_ood.shape[0]/2))
            # writer.add_scalar('ood_metrics/diversity', diversity, i)

            rocauc_ood = ood_metrics_entropy(entropies_test.cpu().detach().numpy(),entropies_ood.cpu().detach().numpy(),writer,config,i,name ='OOD' )
            rocauc_ood_std = ood_metrics_entropy(std_prob_test.cpu().detach().numpy(),std_prob_ood.cpu().detach().numpy(),None,config,i,name ='OOD' )

            writer.add_scalar('ood_metrics/AUROC', rocauc_ood[0], i)
            writer.add_scalar('ood_metrics/AUPR_IN', rocauc_ood[1], i)
            writer.add_scalar('ood_metrics/AUPR_OUT', rocauc_ood[2], i)

            #ECE calculation
            test_labels_np = test_labels.cpu().detach().numpy().astype(np.int8)
            ood_ece = um.numpy.ece(test_labels_ood.cpu().detach().numpy(), average_prob_ood.cpu().detach().numpy(), num_bins=30)
            writer.add_scalar('ood_metrics/ECE', ood_ece, i)

            #bs = um.brier_score(labels=test_labels_np, probabilities=average_prob)
            
            #Confidence histogram with corruption 

            sns.kdeplot(ood_confidence.cpu().detach().numpy(), ax = axes,fill=True, common_norm=False, palette="crest",alpha=.5, linewidth=3,label = 'OOD')
            sns.kdeplot(test_confidence.cpu().detach().numpy(), ax = axes,fill=True, common_norm=False, palette="crest",alpha=.5, linewidth=3,label = 'Test')
            axes.legend()
            writer.add_figure('ood_metrics/confidence', plt.gcf(),i, close=not config.show_plots)
            plt.close()
            test_labels_np = test_labels.cpu().detach().numpy().astype(np.int8)
            test_ece = um.numpy.ece(test_labels_np, Y_t.cpu().detach().numpy(), num_bins=30)
            
            metric = [rocauc_ood[0],rocauc_ood[1],rocauc_ood[2], train_accuracy, test_accuracy, entropies_test.mean().cpu().detach().numpy(), entropies_train.mean().cpu().detach().numpy(), entropies_ood.mean().cpu().detach().numpy(), std_prob_ood.mean().cpu().detach().numpy(), std_prob_test.mean().cpu().detach().numpy(),  test_ece, nll.cpu().detach().numpy(), rocauc_ood_std[0]]
            np.save(config.out_dir+'/'+str(i)+'results', np.array(metric))
            #--------------------------------------------------------------------------------
            # CORRUPTIONS TESTS
            #--------------------------------------------------------------------------------
            corr_test = False
            if corr_test:
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
                ece_l = []
                bs_l = []
                corr_accuracy_l  = []
                roc_auc_l  = []

                for en,x_test_corr in enumerate(blurred):
                        softmax_corr = ensemble.forward(x_test_corr)[0]
                        pred_corr = torch.argmax(softmax_corr,2)

                        average_prob = softmax_corr.mean(0)
                        confidence = torch.max(average_prob,1)[0]

                        #KL_uniform = -torch.sum(average_prob * torch.log2(
                        #    (torch.ones(average_prob.shape[1]) / average_prob.shape[1] )/ average_prob + 1e-20)/log_scale, 1)
                        #KL_uniform[KL_uniform != KL_uniform] = 0
                        #writer.add_scalar('corruption/AV_KL_uniform/'+'inten='+str(intensity[en]), KL_uniform.mean(), i)
                        entropies_corr = -torch.sum(torch.log2(average_prob + 1e-20)/log_scale * average_prob, 1)
                        writer.add_scalar('corruption/Av_entropy/'+'corruption='+str(intensity[en]), entropies_corr.mean(), i)


                        # diversity = torch.mean(torch.min(pred_corr.sum(0), torch.tensor(pred_corr.shape[0])-pred_corr.sum(0)          )/(pred_corr.shape[0]/2))
                        # writer.add_scalar('corr_metrics/diversity', diversity, i)

                        rocauc = ood_metrics_entropy(entropies_test.cpu().detach().numpy(),entropies_corr.cpu().detach().numpy(),writer,config,i,name ='corruption='+str(intensity[en]) )
                        roc_auc_l.append(rocauc[0])
                        writer.add_scalar('corruption/AUROC'+'corruption='+str(intensity[en]), rocauc[0], i)
                        #writer.add_scalar('corruption/AUPR_IN'+'corruption='+str(intensity[en]), rocauc[1], i)
                        #writer.add_scalar('corruption/AUPR_OUT'+'corruption='+str(intensity[en]), rocauc[2], i)

                        #ECE calculation

                        if en==0: 
                            writer.add_scalar('test/ECE', test_ece, i)

                        #bs = um.brier_score(labels=test_labels_np, probabilities=average_prob)
                        corr_accuracy = (torch.argmax(average_prob, 1) == test_labels).sum().item() / Y_t.shape[0]
                        ece_l.append(test_ece)
                        #bs_l.append(bs)
                        corr_accuracy_l.append(corr_accuracy)

                        #Confidence histogram with corruption 

                        #sns.distplot(confidence,hist = False,ax = axes,bins=50,hist_kws={ 'alpha':0.5}, fit_kws={'linewidth':15, 'alpha' :1}, label = 'corr='+str(intensity[en]) )

                        sns.kdeplot(confidence.cpu().detach().numpy(), ax = axes,fill=True, common_norm=False, palette="crest",alpha=.5, linewidth=3,label = 'corr='+str(intensity[en]))
                axes.legend()
                writer.add_figure('corruption/confidence', plt.gcf(),i, close=not config.show_plots)
                plt.close()

                #ECE,bs,accuracy scatter 

                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                ax1.plot(intensity, ece_l, markersize=20, c='b', marker="o", label='ECE')
                #ax1.scatter(bs_l,intensity, s=10, c='r', marker="o", label='Brier')
                ax1.plot(intensity,corr_accuracy_l, markersize=20, c='r', marker="o", label='ACC') 
                ax1.plot(intensity,roc_auc_l, markersize=20, c='orange', marker="o", label='AUROC')
                ax1.legend()


                writer.add_figure('corruption/Scores', plt.gcf(),i, close=not config.show_plots)
                plt.close()
            
            #mem = torch.cuda.memory_allocated(device)
            #print('Test and ood iteration:'+str(i)+' Mem:'+"{:.3f} GB".format(mem / 1024 ** 3), flush = True)


        #if config.keep_samples != 0 and i % config.keep_samples == 0:
        #     samples.append(W.detach().clone())
        #     samples_models = torch.cat(samples)
        #     pred_tensor_samples = ensemble.forward(x_test_0, samples_models)
        #     plot_predictive_distributions(config,writer,i,data, [x_test_0.squeeze()], [pred_tensor_samples.mean(0).squeeze()],
        #                                   [pred_tensor_samples.std(0).squeeze()], save_fig=False, publication_style=False,
        #                                   name=config.method+'smp')

        #if config.save_particles !=0 and i% config.save_particles == 0:
        #    particles = ensemble.particles.cpu().detach().numpy()
        #    np.save(datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.np', particles)

        
            
    return rocauc_ood[0],rocauc_ood[1],rocauc_ood[2], train_accuracy, test_accuracy, entropies_test.mean().cpu().detach().numpy(), entropies_train.mean().cpu().detach().numpy(), entropies_ood.mean().cpu().detach().numpy(), std_prob_ood.mean().cpu().detach().numpy(), std_prob_test.mean().cpu().detach().numpy(),  test_ece, nll.cpu().detach().numpy(), rocauc_ood_std[0]