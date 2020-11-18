import numpy as np

def getMetrics(ground_truth, predictions, positive, negative):
    
    tp = 0;fp = 0; tn = 0; fn = 0;
    for i in range(predictions.shape[0]):
        if ground_truth[i] == positive and predictions[i] == positive:
            tp+=1
        elif ground_truth[i] == positive and predictions[i] == negative:
            fn+=1
        elif ground_truth[i] == negative and predictions[i] == positive:
            fp+=1
        else:
            tn+=1

    return tp,fp,tn,fn






if __name__ == "__main__":
    a = np.array([0,0,0,1,1,1])
    b = np.array([1,1,0,0,1,1])
    tp,fp,tn,fn = getMetrics(a,b,1,0)

    print('tp:{}'.format(tp))
    print('tn:{}'.format(tn))
    print('fp:{}'.format(fp))
    print('fn:{}'.format(fn))

	
