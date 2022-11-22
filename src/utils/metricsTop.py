import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mosi_regression,
            'IEMOCAP': self.__eval_iemocap_classification,
            'MOSEI': self.__eval_mosei_regression,
            'SIMS': self.__eval_sims_regression
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_preds, non_zeros_binary_truth, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_preds, binary_truth, average='weighted')

        eval_results = {
            "Has0_acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_preds_a2, test_truth_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": mult_a2,
            "F1_score": f_score,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "MAE": mae,
            "Corr": corr,  # Correlation Coefficient
        }
        return eval_results

    def __eval_iemocap_classification(self, y_pred, y_true):
        nClass = 4
        _, predicted = torch.max(y_pred.data, 1)

        test_preds = predicted.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        # wa_acc = accuracy_score(test_preds, test_truth)
        # wa_f1 = f1_score(test_preds, test_truth, average='weighted')

        '''
        class_acc = 0
        n_classes = 0
        emo_acc = []
        emo_acc_ang, emo_acc_sad, emo_acc_hap, emo_acc_neu = 0.0, 0.0, 0.0, 0.0
        emo_f1_ang, emo_f1_sad, emo_f1_hap, emo_f1_neu = 0.0, 0.0, 0.0, 0.0
        for c in range(nClass):
            class_pred = np.multiply((test_truth == test_preds),
                                     (test_truth == c)).sum()
            if (test_truth == c).sum() > 0:
                class_pred /= (test_truth == c).sum()
                n_classes += 1
                class_acc += class_pred
        ua_acc = class_acc / n_classes
        ua_f1 = f1_score(test_preds, test_truth, average='micro')
        emo_f1 = f1_score(test_preds, test_truth, average=None)
        emo_f1_1, emo_f1_2, emo_f1_3, emo_f1_4 = emo_f1[0], emo_f1[1], emo_f1[2], emo_f1[3]
        '''
        # emo_acc = []
        # emo_f1 = []

        wa_acc = accuracy_score(test_truth, test_preds)
        ua_acc = recall_score(test_truth, test_preds, average='macro')
        f1 = f1_score(test_truth, test_preds, average='macro')
        # cm = confusion_matrix(test_truth, test_preds)

        # class_acc = 0
        # n_classes = 0
        # for c in range(nClass):
        #     class_pred = np.multiply((test_truth == test_preds),
        #                              (test_truth == c)).sum()
        #     if (test_truth == c).sum() > 0:
        #         class_pred /= (test_truth == c).sum()
        #         n_classes += 1
        #         class_acc += class_pred
        #
        # ua_acc = class_acc / n_classes

        # for c in range(nClass):
        #     emo_acc.append(accuracy_score(test_preds == c, test_truth == c))
        #     emo_f1.append(f1_score(test_preds == c, test_truth == c, average='weighted'))
        # ua_acc = (emo_acc[0] + emo_acc[1] + emo_acc[2] + emo_acc[3]) / 4
        # ua_f1 = (emo_f1[0] + emo_f1[1] + emo_f1[2] + emo_f1[3]) / 4

        eval_results = {
            "wa_acc": wa_acc,
            # "wa_f1": wa_f1,
            "ua_acc": ua_acc,
            "f1": f1,
            # "ua_f1": ua_f1,
            # "ang_acc": emo_acc[0],
            # "ang_f1": emo_f1[0],
            # "sad_acc": emo_acc[1],
            # "sad_f1": emo_f1[1],
            # "hap_acc": emo_acc[2],
            # "hap_f1": emo_f1[2],
            # "neu_acc": emo_acc[3],
            # "neu_f1": emo_f1[3],
        }
        return eval_results

    def getMetics(self, datasetname):
        return self.metrics_dict[datasetname.upper()]