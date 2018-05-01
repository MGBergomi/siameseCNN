import os
import sys
sys.path.append('../utils')
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_utils import *
from input_data_siamese import *
from cnn_utils import *
from siamese_utils import *
import argparse
import h5py
import numpy as np
import pprint

# FIXME: optimal threshold is rounded after saving. FIXED!!!
# IDEA: Use kmeans to choose references.

def cnn_model(x,width, height, channels, keep_prob):
    x_tensor = tf.reshape(x, [-1, width, height, channels])
    # conv1
    filter_size1 = 5
    n_filter1 = 15
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    h_conv1, w1, h1 = buildConv2D('Wconv1','Bconv1',width, height, 1, x_tensor, filter_size1, n_filter1, stride1, pad1)
    # maxpool2d
    stride2 = [1,2,2,1]
    pool2 = 2
    pad2 = 'SAME'
    max_pool2, w2, h2 = maxpool2d(w1,h1, h_conv1, pool2,stride2,pad2)
    d2 = n_filter1
    # conv2
    filter_size3 = 5
    n_filter3 = 50
    stride3 = [1,1,1,1]
    pad3 = 'SAME'
    h_conv3, w3, h3 = buildConv2D('Wconv2','Bconv2',w2, h2, d2, max_pool2, filter_size3, n_filter3, stride3, pad3)
    # maxpool2d
    stride4 = [1,2,2,1]
    pool4 = 2
    pad4 = 'SAME'
    max_pool4, w4, h4 = maxpool2d(w3,h3, h_conv3, pool4,stride4,pad4)
    d4 = n_filter3
    # conv4
    filter_size5 = 5
    n_filter5 = 100
    stride5 = [1,1,1,1]
    pad5 = 'SAME'
    h_conv5, w5, h5 = buildConv2D('Wconv3','Bconv3',w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5

    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    h_conv5_flat = tf.reshape(h_conv5, [-1, resolutionS*d5])
    # fully-connected 1
    n_fc = 100
    h_fc_drop = buildFc('W1','B1',h_conv5_flat,w5,h5,d5,n_fc,1.0)
    h_relu = tf.nn.relu(h_fc_drop)

    n_features = 100
    features = buildFc('W2','B2',h_relu, n_fc,1,1,n_features,1.0)
    h_relu2 = tf.nn.relu6(features)

    n_plotter = 2
    plotter = buildFc('W3','B3',h_relu2, n_features,1,1,n_plotter,1.0)

    return plotter, h_relu2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='matlabexports/imdb_5indiv_3000_1000_s1.mat', type = str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--num_train', default = 2500, type = int)
    parser.add_argument('--num_val', default = 500, type = int)
    parser.add_argument('--num_test', default = 1000, type = int)
    parser.add_argument('--num_epochs', default = 500, type = int)
    parser.add_argument('--itsPerEpoch', default = 100, type = int)
    parser.add_argument('--genuineRatio', default = 0.5, type = float)
    parser.add_argument('--numRefTest', default = 250, type = int)
    args = parser.parse_args()
    path = args.dataset
    num_train = args.num_train
    num_test = args.num_test
    num_valid = args.num_val
    numRefTest = args.numRefTest
    G_ratio = args.genuineRatio

    numIndiv, imsize,\
    X1_train, X1_valid, X1_test,\
    X2_train, X2_valid, X2_test,\
    Y_train, Y_valid, Y_test,\
    Y1_train, Y1_valid, Y1_test,\
    Y2_train, Y2_valid, Y2_test,\
    train_size, totalNumG, totalNumI = dataHelperPlot3(path, num_train, num_test, num_valid, G_ratio)

    resolution = np.prod(imsize)
    x1 = tf.placeholder(tf.float32, [None, resolution])
    x2 = tf.placeholder(tf.float32, [None, resolution])
    yProb = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.float32, [None])
    keep_prob = tf.placeholder(tf.float32)
    numIndivTest = tf.placeholder(tf.int32)
    # share variables
    with tf.variable_scope('convolutions') as scope:
        h1, f1 = cnn_model(x1,imsize[1],imsize[2],imsize[0],keep_prob)
        scope.reuse_variables()
        h2, f2 = cnn_model(x2,imsize[1],imsize[2],imsize[0],keep_prob)
    # get loss function
    lossFunc = contrastive_loss1(f1, f2, y)
    # compute the mean
    contrastiveLoss = tf.reduce_mean(lossFunc)
    # for training with SGD or SGD with momentum we define a decaying learning rate
    learning_rate = tf.placeholder(tf.float32, shape=[])
    lr0 = 0.001
    gamma = 0.001
    power = 0.75
    # collection of tested optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(contrastiveLoss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(contrastiveLoss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(contrastiveLoss)
    # Monitor accuracy
    FPR, TPR, acc, optimalThInd, ths = computeROCAccuracy(f1, f2, y)
    optimal_threshold = tf.Variable(0., name='optimal_threshold', trainable=False)
    # idTracker accuracy
    featProb, featRef = getIds(f1,f2,yProb,optimal_threshold,numIndivTest) # f1-> featProb, f2-> featRef
    ckpt_dir = args.ckpt_folder
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()
    # Run
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
        start = global_step.eval() # get last global_step
        print("Start from:", start)
        # We'll now train in minibatches and report accuracy, loss:
        iter_per_epoch = args.itsPerEpoch
        n_epochs = args.num_epochs
        indices = np.linspace(0, train_size, iter_per_epoch)
        indices = indices.astype('int')
        plt.ion()
        if args.train == 1:
            if start == 0:
                ###COMPUTE FEATURES, LOSS AND ACCURACY FOR TRAIN AND VAL BEFORE TRAINING
                print('*** Computing features loss and accuracy for the first time...')
                print('-Training set-')
                clusterPlotTrain = {'h1':[],'h2':[],'f1':[],'f2':[]}
                lossEpoch = []
                accEpoch = []
                npFPRtrain = []
                npTPRtrain = []
                print(' Entering loop for batches')
                for iter_i in range(iter_per_epoch - 1):
                    batch_x1s = X1_train[indices[iter_i]:indices[iter_i+1]]
                    batch_x2s = X2_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
                    feat1, feat2,hdFeat1,hdFeat2 = sess.run([h1,h2,f1,f2],
                                    feed_dict={
                                        x1: batch_x1s,
                                        x2: batch_x2s,
                                        keep_prob: 1.
                                    })
                    if iter_i % 50 == 0:
                        loss, _, _, ROCAccTrain = sess.run([contrastiveLoss, FPR, TPR, acc],
                                        feed_dict={
                                            x1: batch_x1s,
                                            x2: batch_x2s,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        print("  Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.10f}".format(loss) + ", Minibatch Acc= " + "{:.10f}".format(ROCAccTrain))
                        lossEpoch.append(loss)
                        accEpoch.append(ROCAccTrain)

                    clusterPlotTrain['h1'].append(feat1)
                    clusterPlotTrain['h2'].append(feat2)
                    clusterPlotTrain['f1'].append(hdFeat1)
                    clusterPlotTrain['f2'].append(hdFeat2)

                print( Concatenating minibatches... ')
                clusterPlotTrain['h1'] = [inner for outer in clusterPlotTrain['h1'] for inner in outer]
                clusterPlotTrain['h2'] = [inner for outer in clusterPlotTrain['h2'] for inner in outer]
                clusterPlotTrain['f1'] = [inner for outer in clusterPlotTrain['f1'] for inner in outer]
                clusterPlotTrain['f2'] = [inner for outer in clusterPlotTrain['f2'] for inner in outer]
                lossTrainPlot = [np.mean(lossEpoch)]
                accTrainPlot = [np.mean(accEpoch)]
                print('-Validation set-')
                feat1val, feat2val = sess.run([h1,h2],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    keep_prob: 1.
                                })
                clusterPlotVal = {'h1':feat1val,'h2':feat2val}

                lossVal, FPRVal, TPRVal, ROCAccVal, optThInd, Th = sess.run([contrastiveLoss, FPR, TPR, acc, optimalThInd, ths],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    y: Y_valid,
                                    keep_prob: 1.0
                                })
                lossValPlot = [lossVal]
                accValPlot = [ROCAccVal]
                print("  No training yet , Validation Loss= " + "{:.10f}".format(lossVal) + ", Validation Acc= " + "{:.10f}".format(ROCAccVal))
                ### plotting
                plotterSiameseAccuracy(
                    lossTrainPlot, lossValPlot,
                    accTrainPlot, accValPlot,
                    TPRVal, FPRVal,
                    clusterPlotTrain, clusterPlotVal,
                    Y1_train, Y1_valid,
                    Y2_train, Y2_valid,
                    numIndiv)
                print('Saving figure...')
                figname = ckpt_dir + '/result_start.pdf'
                plt.savefig(figname)

            ###TRAINING
            print('*** Starting training '))
            if start != 0:
                lossTrainPlot = []
                accTrainPlot = []
                lossValPlot = []
                accValPlot = []

            lr = lr0
            for epoch_i in range(n_epochs):
                clusterPlotTrain = {'h1':[],'h2':[],'f1':[],'f2':[]}
                lossEpoch = []
                accEpoch = []
                for iter_i in range(iter_per_epoch - 1):
                    batch_x1s = X1_train[indices[iter_i]:indices[iter_i+1]]
                    batch_x2s = X2_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
                    feat1, feat2,hdFeat1,hdFeat2 = sess.run([h1,h2,f1,f2],
                                    feed_dict={
                                        x1: batch_x1s,
                                        x2: batch_x2s,
                                        keep_prob: 1.
                                    })
                    clusterPlotTrain['h1'].append(feat1)
                    clusterPlotTrain['h2'].append(feat2)
                    clusterPlotTrain['f1'].append(hdFeat1)
                    clusterPlotTrain['f2'].append(hdFeat2)
                    if iter_i % 50 == 0:
                        loss, _, _, ROCAccTrain = sess.run([contrastiveLoss, FPR, TPR, acc],
                                        feed_dict={
                                            x1: batch_x1s,
                                            x2: batch_x2s,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        print("  Epoch " + str(epoch_i) + ", iter " + str(iter_i) + ", Minibatch Loss= " + "{:.10f}".format(loss) + ", Minibatch Acc= " + "{:.10f}".format(ROCAccTrain))
                        lossEpoch.append(loss)
                        accEpoch.append(ROCAccTrain)
                        maxDist = np.max(np.sum(np.abs(np.subtract(hdFeat1,hdFeat2)),axis=1))
                    # lr = lr0 * (1 + gamma * epoch_i) ** -power
                    # optimizer.lr = args.lr * (1 + args.gamma * epoch) ** -args.power
                    sess.run(optimizer,
                            feed_dict={
                                x1: batch_x1s,
                                x2: batch_x2s,
                                y: batch_ys,
                                keep_prob: 1.,
                                # learning_rate: lr
                            })
                lossTrainPlot.append(np.mean(lossEpoch))
                accTrainPlot.append(np.mean(accEpoch))
                clusterPlotTrain['h1'] = [inner for outer in clusterPlotTrain['h1'] for inner in outer]
                clusterPlotTrain['h2'] = [inner for outer in clusterPlotTrain['h2'] for inner in outer]
                clusterPlotTrain['f1'] = [inner for outer in clusterPlotTrain['f1'] for inner in outer]
                clusterPlotTrain['f2'] = [inner for outer in clusterPlotTrain['f2'] for inner in outer]
                print('Computing loss, accuracy and feautres for the validation set...')
                feat1val, feat2val = sess.run([h1,h2],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    keep_prob: 1.
                                })
                clusterPlotVal = {'h1':feat1val,'h2':feat2val}
                lossVal, FPRVal, TPRVal, ROCAccVal, optThInd, Th = sess.run([contrastiveLoss, FPR, TPR, acc, optimalThInd, ths],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    y: Y_valid,
                                    keep_prob: 1.0
                                })
                lossValPlot.append(lossVal)
                accValPlot.append(ROCAccVal)
                print("Epoch " + str(epoch_i) + ", Validation Loss= " + "{:.10f}".format(lossVal) + ", Validation Acc= " + "{:.10f}".format(ROCAccVal))
                print("Optimal Th " + str(Th[optThInd]))
                ###plotting
                plotterSiameseAccuracy(
                    lossTrainPlot, lossValPlot,
                    accTrainPlot, accValPlot,
                    TPRVal, FPRVal,
                    clusterPlotTrain, clusterPlotVal,
                    Y1_train, Y1_valid,
                    Y2_train, Y2_valid,
                    numIndiv)
                print('Saving model...')
                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                optimal_threshold.assign(Th[optThInd]).eval()
                saver.save(sess, ckpt_dir + "/model.ckpt",global_step=global_step)
                print('Saving figure...')
                figname = ckpt_dir + '/result_' + str(epoch_i) + '.pdf'
                plt.savefig(figname)
                print('-------------------------------')
        if args.train == 0:
            featTest = sess.run(h1,
                            feed_dict={
                                x1: X1_valid,
                                keep_prob: 1.
                            })
            plotterSiameseTest(featTest,Y1_valid,numIndiv)
            plt.draw()
        if args.train == 2:
            print("Computing accuracy in IdTracker style")
            opTh = sess.run(optimal_threshold)
            print("Optimal Threshold:", opTh)
            refImages, probImages, refLabels, probLabels = PrepareDataTest(numRefTest, X1_test, X2_test, Y1_test, Y2_test)
            featP, featR = sess.run([featProb, featRef],
                            feed_dict={
                                x1: probImages,
                                x2: refImages,
                                yProb: probLabels,
                                optimal_threshold: opTh,
                                numIndivTest: numIndiv # this numIndiv can be different in the future because the number of animals in the training may be different than in the test
                            })
            # compute the number of individuals in test
            indivProb = list(set(probLabels))
            numIndivProb = len(indivProb)
            indivRef = list(set(refLabels))
            numIndivRef = len(indivRef)
            print("individuals problem images: " +  str(countRateSet(probLabels)))
            print("individuals reference images: " +  str(countRateSet(refLabels)))
            indivAcc = np.zeros(numIndivProb)
            failCollector = []
            plotFails = []
            numIndivPerClassProb = [countRateSet(probLabels)[i][1] for i in range(len(countRateSet(probLabels)))]
            numProbImages = len(probLabels)
            ### Perform reference's clusters analysis to discard outliers
            # import KNN
            from KNN import kMeansCluster
            # initalize variables to store centroids
            newFeatRef = []
            newLabRef = []
            for i in range(numIndivRef):
                # get the indices associated to the ith individual
                inds_i = [np.asarray(refLabels) == i]
                # and thus its features and labels in the reference set
                # REMARK: features are organised as follows
                # [numOfFeatures, dimensionality]
                refFeat_i = featR[inds_i]
                refLab_i = np.asarray(refLabels)[inds_i]
                # print refFeat_i.shape
                numClusterPerInd = 25
                centroid_values, assignment_values = kMeansCluster(refFeat_i, numClusterPerInd, 100, stop_coeficient = 0.0)
                # associate a label to centroids
                centLab = refLab_i[:numClusterPerInd]
                newFeatRef.append(centroid_values)
                newLabRef.append(centLab)
            newFeatRef = flatten(newFeatRef)
            newLabRef = flatten(newLabRef)
            # set counter for correct predictions
            c = 0
            for ind in range(numProbImages):
                featP_ = featP[ind]
                labP_ =  probLabels[ind]
                match = []
                for ref in newFeatRef:
                    # compute the distance (L1)
                    dist = np.sum(np.abs(np.subtract(featP_, ref)))
                    if dist <=  opTh:
                        match.append(True)
                    else:
                        match.append(False)
                matchedRefs = np.where(match)
                counter = [newLabRef[int(i)] for i in matchedRefs[0]]
                IdDistrib = countRate(counter)
                # sorted Id distribution
                sorted_IdDistrib = sorted(IdDistrib, key=lambda tup: tup[1], reverse = True)
                # prediction
                if len(sorted_IdDistrib) > 0:
                    predictedID = sorted_IdDistrib[0][0]
                else:
                    predictedID = 10000
                if predictedID == labP_:
                    c += 1
                    indivAcc[int(labP_)] += 1
                else:
                    failCollector.append([str(sorted_IdDistrib), "prediction: " + str(predictedID), "ground truth: " + str(labP_)])
                    plotFails.append([sorted_IdDistrib, predictedID, labP_])
                # compute individual accuracy
                indivInd = int(labP_)
            print("----------------------------------------------")
            print("number of problem images --------------> " + str(numProbImages))
            print("number of correct assignments ---------> " + str(c))
            print("average number of fails per individual > " + str(np.true_divide(numProbImages - c , numIndiv)))
            print("----------------------------------------------")
            accuracy = np.true_divide(c, numProbImages)
            print("overall accuracy ----------------------> " + str(accuracy))
            indivAccuracy = np.true_divide(indivAcc, numIndivPerClassProb)
            print(indivAccuracy)
