# By Namli & Nouri (13.04.2022)

# required libraries are imported
import datetime as dt
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import auc
from itertools import cycle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score,balanced_accuracy_score

n0 = dt.datetime.now()

# determine the values of the required parameters
pieces_testset = 10  # define the number of pieces for testset
sinifsayisi = 2  # define the # of class
itrsayisi = 100  # define the # of iteration for training
adimbuyuklugusabiti = 0.8  # define the step size constant
epsilon = 0.1  # define the epsilon for stop criteria (mean(e_ani) < epsilon)
alfa = 0.7  # define the alpha value for momentum term
Similar = [0.6, 0.5, 0.4]
percentage = 60
featureselectionnumber = 10
Selected_Feature_Number = 10
durum = 0

# determine the values of the required parameters for SOM
neuronnumber = 120
gridSizeX = 10
gridSizeY = 10
iterationnumber = 100
sinifsayisi = 2
kosturmasayisi = 1
sigma_I = 3
landa = 2.5
L_I = 1

# define necessary variables
E_iterasyon = []
E_iterasyon_test = []
egitim_itr_sayisi = []
Accuracy_List = []
Accuracy_yavas_List = []
Precision_List = []
Recall_List = []
F_Measure_List = []
roc_auc_ovo_List=[]
roc_auc_ovr_List=[]
Specificity_List=[]
Balanced_accuracy_List=[]
specificity_for_class_1_List=[]
specificity_for_class_0_List=[]
EgitimSure = []
TestSure = []
TestSureSOM = []
EgitimSureSOM = []
Train_Index_Tut = []
W_son_Tut = []
c_son_Tut = []
sigma_son_Tut = []
TestSetS = []
Features_index_const = []
Potentialindex_ALL = []
Potentialpredict_ALL = []
predict_ALL = []

# Import Data set
df = pd.read_csv('adult_ql.csv')  # (use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xlsx'
Input = df.to_numpy()  # Convert Data set to numpy(Data set include Feacture ID,Input Index and Class information)
numberofinput = len(Input[1:, :])  # there is how many number of sample in data set


# Define percentage function
def calc_percentage(a, b):
    return round((a * b) / 100, 0)


# Define function to split data set to training and test data set (20% for training rest for test)
def Generate_Training_Test_Set(Input):
    Numberof_Training_Set = int(
        calc_percentage(percentage, numberofinput))  # find number of 20% of data set for training
    Input_Index = list(range(1, (numberofinput + 1)))  # list of input index
    Random_Training_Index = np.array(random.sample(Input_Index,
                                                   Numberof_Training_Set))  # select random 20% for training set(according to Input Index)
    Random_Training_Index.sort()

    # Creat Training  set
    Training_Set1 = np.array(Input[Random_Training_Index.astype(
        int)])  # Training set without features ID(Create  Training Set according to random selected Index)
    Training_Set = np.vstack((Input[0, :], Training_Set1))  # Add features ID to Training set

    # Creat Test set
    Test_Set = np.array(np.delete(Input, Random_Training_Index,
                                  axis=0))  # Delete selected training set from input to create Test set (without Features ID)
    Test_Set_Index=list(Test_Set[1:,0]) #Test set Index
    NumberofSample_TestSet = len(Test_Set)
    return Training_Set, Random_Training_Index, Test_Set, Test_Set_Index, NumberofSample_TestSet


def Feature_Selection(Features_index_const, IN):
    if len(Features_index_const) == 0:
        Features = list(np.unique(IN[0, 1:-sinifsayisi]))  # find the list unique features
        Random_Features = np.array(
            random.sample(Features, Selected_Feature_Number))  # select 10 number of features randomly
        Random_Features.sort()

        SelecetedFeatures_index = []
        for h in Random_Features:
            index = np.argwhere(IN[0, 1:-sinifsayisi] == h)
            index = np.reshape(index, (1, len(index)))  # find the index of each randomly selected features
            SelecetedFeatures_index = np.append(SelecetedFeatures_index, index[0])
            SelecetedFeatures_index = SelecetedFeatures_index.astype(int)
    else:
        SelecetedFeatures_index = Features_index_const

    IN1 = IN[:, 1:-sinifsayisi]
    TestSet_With_Selectedindex1 = IN1[:,
                                  SelecetedFeatures_index]  # Create test set with random selected index ( without Input Index and class information with Features ID)
    TestSet_With_Selectedindex2 = np.vstack((TestSet_With_Selectedindex1.T, IN[:,
                                                                            -sinifsayisi:].T))  ## add  class information to last column of test with random selected index (data set without Input  Index and with class information and Feature ID)
    TestSet_With_Selectedindex3 = np.vstack((IN[:, 0].T,
                                             TestSet_With_Selectedindex2))  ## add Input Index to first column of test set(data set with selected features with Input  index and with class information and Feature ID)
    TestSet_With_Selectedindex = TestSet_With_Selectedindex3.T  # transpose
    dim_Som = len(SelecetedFeatures_index)
    return SelecetedFeatures_index, TestSet_With_Selectedindex, dim_Som


# define the fi function to construct the fi matrix
def fi(x, c_son, sigma_son, fi_n):
    fimatris = []
    for i in range(0, fi_n):
        fimatris += [np.exp(-((distance.euclidean(x, c_son[i, :])) ** 2) / (2 * sigma_son[0, i] ** 2))]
    fimatris = np.array(fimatris)
    return (fimatris)


# derivative of fi-c-matris
def derivfic(x, c_son, sigma_son, fi_n):
    derivficmatris = []
    for i in range(0, fi_n):
        derivficmatris += [((x - c_son[i, :]) / (sigma_son[0, i] ** 2)) * (
            np.exp(-((distance.euclidean(x, c_son[i, :])) ** 2) / (2 * sigma_son[0, i] ** 2)))]
    derivficmatris = np.array(derivficmatris)
    return (derivficmatris)


# derivative of fi-sigma
def derivfisigma(x, c_son, sigma_son, fi_n):
    derivfimatris = []
    for i in range(0, fi_n):
        derivfimatris += [((distance.euclidean(x, c_son[i, :])) ** 2) / sigma_son[0, i] ** 3 * (
            np.exp(-((distance.euclidean(x, c_son[i, :])) ** 2) / (2 * sigma_son[0, i] ** 2)))]
    derivfimatris = np.array(derivfimatris)
    return (derivfimatris)


# define the Wturet function to calculate weight matrix
def Wturet(WListe, e, fi_n, Inputcolumn):
    Wturet = []
    turet = np.dot(WListe[1].T, e.reshape(sinifsayisi, 1))
    for i in range(0, Inputcolumn):
        Wturet += [turet]
    Wturet = np.array(Wturet)
    return ((Wturet.T).reshape(fi_n, Inputcolumn))


# Location of neurons for SOM
Neuron_XY = np.zeros((neuronnumber, 2))
for i in range(neuronnumber):
    s = divmod(i, gridSizeX)
    Neuron_XY[i][0] = s[1] + 1
    Neuron_XY[i][1] = s[0] + 1

# distance between nerouns location for SOM
d_ij = np.zeros((neuronnumber, neuronnumber))
for i in range(0, neuronnumber):
    for j in range(0, neuronnumber):
        d_ij[i][j] = distance.euclidean(Neuron_XY[i,], Neuron_XY[j,])  # distance between nerouns


# training phase of RBFN:
def rbfn_training(Inputmatrix, Outputmatrix):
    n1 = dt.datetime.now()

    # RF# define new dataset dimension
    Inputcolumn = len(Inputmatrix[0])
    fi_n = Inputcolumn + 1

    # define the matrix consisting of c vectors
    ceski = [[round(random.uniform(0.1, 0.9), 2) for x in range(Inputcolumn)] for y in range(fi_n)]
    cyeni = [[round(random.uniform(0.1, 0.9), 2) for x in range(Inputcolumn)] for y in range(fi_n)]
    cListe = [np.array(ceski), np.array(cyeni)]

    # define the array containing the sigma values
    sigmaeski = [[round(random.uniform(0.5, 1.5), 2) for x in range(fi_n)] for y in range(1)]
    sigmayeni = [[round(random.uniform(0.5, 1.5), 2) for x in range(fi_n)] for y in range(1)]
    sigmaListe = [np.array(sigmaeski), np.array(sigmayeni)]

    # create weight matrices that will be used for the momentum term
    Weski = [[round(random.uniform(0.01, 0.25), 2) for x in range(fi_n)] for y in range(sinifsayisi)]
    Wyeni = [[round(random.uniform(0.01, 0.25), 2) for x in range(fi_n)] for y in range(sinifsayisi)]
    WListe = [np.array(Weski), np.array(Wyeni)]

    E = []
    W_son = []

    for j in range(0, itrsayisi):

        # define step sizes to update weights
        adimbuyuklugu = adimbuyuklugusabiti - j * (0.60 / itrsayisi)

        E_ani = []

        for i in range(0, len(Inputmatrix)):
            # calculate output value
            Y = np.dot(WListe[1], fi(Inputmatrix[i, :], cListe[1], sigmaListe[1], fi_n))
            # calculate error term
            e = Outputmatrix[i, :] - Y
            E_ani.append(0.5 * np.dot(e.T, e))

            # keep the weights before update
            W_tut = WListe[1]
            c_tut = cListe[1]
            sigma_tut = sigmaListe[1]
            # keep updated weights as second item of list
            WListe[1] = WListe[1] + adimbuyuklugu * np.dot(e.reshape(sinifsayisi, 1),
                                                           fi(Inputmatrix[i, :], cListe[1], sigmaListe[1],
                                                              fi_n).reshape(1, fi_n)) + alfa * (WListe[1] - WListe[0])
            cListe[1] = cListe[1] + adimbuyuklugu * Wturet(WListe, e, fi_n, Inputcolumn) * derivfic(Inputmatrix[i, :],
                                                                                                    cListe[1],
                                                                                                    sigmaListe[1],
                                                                                                    fi_n) + alfa * (
                                    cListe[1] - cListe[0])
            sigmaListe[1] = sigmaListe[1] + adimbuyuklugu * np.dot(WListe[1].T, e.reshape(sinifsayisi, 1)) * (
                derivfisigma(Inputmatrix[i, :], cListe[1], sigmaListe[1], fi_n).reshape(fi_n, 1)) + alfa * (
                                        sigmaListe[1] - sigmaListe[0])
            # keep the weights of one step before as the first item in the list
            WListe[0] = W_tut
            cListe[0] = c_tut
            sigmaListe[0] = sigma_tut

        E.append(np.mean(E_ani))

        # calculate the mean error for the relevant iteration
        if (np.mean(E_ani) < epsilon):
            W_son = np.array(WListe[1])
            c_son = np.array(cListe[1])
            sigma_son = np.array(sigmaListe[1])
            break

    egitim_itr_sayisi.append(len(E))
    E_ani_egitim_deger = np.mean(E)
    # keep the latest updated weight matrix
    if len(W_son) == 0:
        W_son = np.array(WListe[1])
        c_son = np.array(cListe[1])
        sigma_son = np.array(sigmaListe[1])

    n2 = dt.datetime.now()
    return W_son, c_son, sigma_son, E_ani_egitim_deger, egitim_itr_sayisi, n2, n1


# test phase of RBFN:
def rbfn_test(Inputtestmatrix, Outputtestmatrix, W_son, c_son, sigma_son):
    # print("Inputtestmatrix",Inputtestmatrix)
    n3 = dt.datetime.now()
    E_ani_test = []
    Yd_hesaplanan = []
    Test_Yd_hesaplanan_Sinif = []

    Inputtestrow = len(Inputtestmatrix)
    Outputtestrow = len(Outputtestmatrix)
    Inputtestcolumn = len(Inputtestmatrix[0])
    fi_n = Inputtestcolumn + 1

    for i in range(0, Inputtestrow):
        # calculate output value
        Ytest = np.dot(W_son, fi(Inputtestmatrix[i, :], c_son, sigma_son, fi_n))

        for j in range(0, sinifsayisi):
            if Ytest[j] >= sum(Ytest) - Ytest[j]:
                Ytest[j] = 1
            else:
                Ytest[j] = 0

        # calculate error term
        etest = Outputtestmatrix[i, :] - Ytest
        E_ani_test.append(0.5 * np.dot(etest.T, etest))
        Yd_hesaplanan.append(Ytest)
    E_ani_test_deger = np.mean(E_ani_test)

    for i in range(0, Outputtestrow):
        if sum(Yd_hesaplanan[i]) == 1:
            for j in range(0, sinifsayisi):
                if Yd_hesaplanan[i][j] == 1:
                    Test_Yd_hesaplanan_Sinif.append(j)
        else:
            Test_Yd_hesaplanan_Sinif.append(sinifsayisi)
    n4 = dt.datetime.now()
    return Test_Yd_hesaplanan_Sinif


def SOM_training(neuronnumber, Input, dim_Som):
    Inputclass = []
    numberofsample = len(Input[0])
    Output=Input[1:,-sinifsayisi:]
    for k in range(0, numberofsample):
        for l in range(0, sinifsayisi):
            if Output[k][l] == 1:
                Inputclass.append(l)

    # Inputclass=Input[1:,-1]
    Input = Input[1:, 1:-sinifsayisi].T

    Egtim_suresi_1 = dt.datetime.now()
    w = np.array([[round(random.uniform(-0.25, 0.25), 4) for x in range(neuronnumber)] for y in
                  range(dim_Som)])  ##Initializing The Weights
    w_initial = w
    Winners_old = np.zeros(numberofsample)
    for n in range(1, iterationnumber + 1):  # beginning of each iteration
        Winners = np.zeros(numberofsample)
        for a in range(0, numberofsample):
            Distance = np.zeros((neuronnumber, numberofsample))
            for b in range(0, neuronnumber):
                Distance[b][a] = distance.euclidean(Input[:, a], w[:,
                                                                 b])  # calculate distance between each input and neurons by euclidean norm
            mindist = min(Distance[:, a])  # find minimum distance
            Winner = random.choice(np.argwhere(Distance[:, a] == np.min(Distance[:, a])))  # find index of winner neuron
            Winners[a] = Winner  # put winners in one vector for each input
            # Start updating
            sigma_N = sigma_I * np.exp(-n / (landa))  # the exponential decay function)
            H_n = np.exp(-(d_ij[Winner, :]) ** 2 / (2 * (sigma_N) ** 2))  # disp('H_n is the neighborhood function')
            H_nn = np.ones((dim_Som, 1)) * H_n
            L_n = L_I * np.exp(-n / landa)  # the learning rate parameter
            w_N = (w + (L_n * H_nn * (Input[:, a].reshape(dim_Som, 1) - w))).reshape(dim_Som,
                                                                                     neuronnumber)  # disp('w_N Is the updated weight vector')
            w_old = w  # keep old weights in w_old
            w = w_N  # replace old weights with updated weights
        # if (Winners_old == Winners).all():  # if winner in this step will be same as the previous step "Winners are same" will be printed
        # print("Winners are same")
        Winners_old = Winners
    Egtim_suresi_2 = dt.datetime.now()
    som_sinifbilgisi = np.zeros(numberofsample)
    obeklere_giren_veri_sayisi = np.zeros(neuronnumber)
    kac_sinifdan_veri = np.zeros(neuronnumber)
    obek_etiketi = np.zeros(neuronnumber)
    silinen_agliklar = []
    for l in range(0, neuronnumber):
        globals()[str(l + 1) + "obek"] = np.where(Winners == l)  # Test de l.obege hangi veriler var
        obeklere_giren_veri_sayisi[l] = np.sum(Winners == l)
        globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"] = np.zeros(
            len(globals()[str(l + 1) + "obek"][0]))
        if (obeklere_giren_veri_sayisi[l] != 0):
            for m in range(0, len(globals()[str(l + 1) + "obek"][0])):
                globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"][m] = Inputclass[
                    globals()[str(l + 1) + "obek"][0][m]]
            globals()[str(l + 1) + ".uniqueValues_sinifbilgisi"], occurCount_sinifbilgisi_test = np.unique(
                globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"], return_counts=True)
            kac_sinifdan_veri[l] = globals()[str(l + 1) + ".obekde kac sinfdan veri var"] = len(
                np.unique(globals()[str(l + 1) + ".obege giren verilerin sinif bilgisi"]))
            for v in range(0, sinifsayisi):
                if (globals()[str(l + 1) + ".uniqueValues_sinifbilgisi"] == v).all():
                    som_sinifbilgisi[globals()[str(l + 1) + "obek"]] = v
                    obek_etiketi[l] = v
                elif len(globals()[str(l + 1) + ".uniqueValues_sinifbilgisi"]) > 1:
                    som_sinifbilgisi[globals()[str(l + 1) + "obek"]] = sinifsayisi
                    obek_etiketi[l] = sinifsayisi
        else:
            kac_sinifdan_veri[l] = 0
            silinen_agliklar.append(l)  # index of clusters which are empty
            obek_etiketi[l] = sinifsayisi + 1
    EgitimSureSOM.append(Egtim_suresi_2 - Egtim_suresi_1)
    silinen_agliklar = np.array(silinen_agliklar)
    w_test = np.delete(w, silinen_agliklar, axis=1)  # delet weights which are not win
    obek_etiketi = obek_etiketi[obek_etiketi != sinifsayisi + 1]  # delet empty clusters
    neuronnumber_test = neuronnumber - len(
        silinen_agliklar)  # delet number of neuron whıch are not win from whole number of neuron
    return w_test, Winners, EgitimSureSOM, som_sinifbilgisi, obek_etiketi, silinen_agliklar, neuronnumber_test


def SOM_test(neuronnumber_test, Input_test, w_test, obek_etiketi):
    Input_test = Input_test[1:, :].T
    numberofinput_test = len(Input_test[0])
    Test_suresi_1 = dt.datetime.now()
    Winners_test = np.zeros(numberofinput_test)
    sinifbilgisi_test = np.zeros(numberofinput_test)
    for a in range(0, numberofinput_test):

        Distance_test = np.zeros((neuronnumber_test, numberofinput_test))
        for b in range(0, neuronnumber_test):
            Distance_test[b][a] = distance.euclidean(Input_test[:, a], w_test[:,
                                                                       b])  # calculate distance between each input and neurons by euclidean norm
        mindist_test = min(Distance_test[:, a])  # find minimum distance
        Winner_test = random.choice(np.argwhere(Distance_test[:, a] == np.min(Distance_test[:, a])))
        for v in range(0, sinifsayisi):
            sinifbilgisi_test[a] = obek_etiketi[Winner_test]
        Winners_test[a] = Winner_test  # put wiiners in one vector for each input
    Test_suresi_2 = dt.datetime.now()
    TestSureSOM.append(Test_suresi_2 - Test_suresi_1)
    return sinifbilgisi_test


# calculate performance criteria
def performance(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif):
    confusionmatrix=confusion_matrix(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif)
    tp1, fn1, fp1, tn1 = confusion_matrix(Test_Yd_Sinif, Test_Yd_hesaplanan_Sinif).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(Test_Yd_Sinif, Test_Yd_hesaplanan_Sinif).ravel()
    accuracy = round(accuracy_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif), 3)
    print("confusionmatrix",confusionmatrix)
    # Calculate accuracy for "yavas" class.
    sinifsayisinihai=sinifsayisi+1
    ConfusionMatrixforYavas = np.array([[sum([(Test_Yd_Sinif[i] == true_class) and (Test_Yd_hesaplanan_Sinif[i] == pred_class)
                                      for i in range(len(Test_Yd_Sinif))])
                                 for pred_class in range(0, sinifsayisinihai)]
                                for true_class in range(0, sinifsayisinihai)])
    accuracy_yavas = round(ConfusionMatrixforYavas[1][1] / np.sum(ConfusionMatrixforYavas, axis=1)[1], 3)
    # Calculate accuracy for "yavas" class.
    specificity_for_class_1 = round(tn1 / (tn1 + fp1), 3)
    specificity_for_class_0 = round(tn2 / (tn2 + fp2), 3)
    specificity=round((specificity_for_class_1 + specificity_for_class_0)/2, 3)
    balanced_accuracy = round(balanced_accuracy_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif), 3)
    recall = round(recall_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif,average="macro"), 3)
    precision = round(precision_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif,average="macro"), 3)
    fmeasure = round(f1_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif,average="macro"), 3)
    weighted_roc_auc_ovo = round(roc_auc_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif, multi_class="ovo", average="weighted"), 3)
    weighted_roc_auc_ovr = round(roc_auc_score(Test_Yd_Sinif,Test_Yd_hesaplanan_Sinif, multi_class="ovr", average="weighted"), 3)
    return accuracy, accuracy_yavas, recall, precision, fmeasure, confusionmatrix, weighted_roc_auc_ovo, weighted_roc_auc_ovr, specificity, balanced_accuracy,specificity_for_class_1,specificity_for_class_0



def accuracycheck(accuracy, accuracy_yavas):
    mevcutdurum = 0
    if accuracy >= 0.80:
        if accuracy_yavas >= 0.70:
            mevcutdurum = 1
    else:
        mevcutdurum = 0
    return mevcutdurum


# FLOW #
Training_Set, Random_Training_Index, Test_Set, Test_Set_Index, NumberofSample_TestSet = Generate_Training_Test_Set(
    Input)

# divide trainingset into two different sub-set
samplesize = round(len(Training_Set) / 2)
Training_Set_SOM = Training_Set[:samplesize, :]
Training_Set_RBFN = Training_Set[samplesize:, :]

# divide testset into n (pieces_testset) different sub-set
X = round(NumberofSample_TestSet / pieces_testset)  # number of sample after cute test set to 10 pieces
kucukx = 1
for i in range(0, pieces_testset):
    A = np.vstack((Test_Set[0, :], Test_Set[kucukx:kucukx + X, :]))
    TestSetS.append(A)
    kucukx = kucukx + X
TestSetS.append(Test_Set[kucukx:, :])

# train the model
W_son, c_son, sigma_son, E_ani_egitim_deger, egitim_itr_sayisi, n2, n1 = rbfn_training( Training_Set_RBFN[:, 1:-sinifsayisi], Training_Set_RBFN[:, -sinifsayisi:])

for f in range(0, pieces_testset):
    print(f, ". döngü:")
    nf1 = dt.datetime.now()
    featureselectioncount = 0
    durum = 0

    while durum == 0:
        SelecetedFeatures_index, TrainSet_With_Selectedindex, dim_Som = Feature_Selection(Features_index_const,
                                                                                          Training_Set_SOM)
        SelecetedFeatures_index, TestSet_With_Selectedindex, dim_Som = Feature_Selection(SelecetedFeatures_index,
                                                                                         TestSetS[f])
        # print("TestSetS[f]",TestSetS[f])
        # print("TestSet_With_Selectedindex", TestSet_With_Selectedindex)
        if len(Features_index_const) == 0:
            w_test, Winners, EgitimSureSOM, som_sinifbilgisi, obek_etiketi, silinen_agliklar, neuronnumber_test = SOM_training(neuronnumber, TrainSet_With_Selectedindex, dim_Som)
        Test_Yd_SOM = SOM_test(neuronnumber_test, TestSet_With_Selectedindex[:, 1:-sinifsayisi], w_test, obek_etiketi)
        Potential_Yavas = []
        for k in range(0, len(Test_Yd_SOM)):
            if Test_Yd_SOM[k] == sinifsayisi - 1:
                Potential_Yavas.append(TestSet_With_Selectedindex[k + 1, 0])
        Potentialindex = np.array(Potential_Yavas)
        # print("Potentialindex",Potentialindex)
        Seleceted_index = []
        Potential_Yavas_TestSet = []
        if len(Potentialindex) == 0:
            print("Potentialindex BOS GELDİ!!!")
        else:
            for h in Potentialindex:
                index = np.argwhere(Test_Set[:, 0] == h)
                index = np.reshape(index, (1, len(index)))  # find the index of each randomly selected features
                Seleceted_index = np.append(Seleceted_index, index[0])
                Seleceted_index = Seleceted_index.astype(int)
            Potential_Yavas_TestSet = Test_Set[Seleceted_index, :]
            Test_Yd_hesaplanan_Sinif = rbfn_test(Potential_Yavas_TestSet[:, 1:-sinifsayisi],
                                                 Potential_Yavas_TestSet[:, -sinifsayisi:], W_son, c_son, sigma_son)
            Potential_Yavas_TestSet_Reel = []
            for k in range(0, len(Potential_Yavas_TestSet)):
                for l in range(0, sinifsayisi):
                    if Potential_Yavas_TestSet[k][-sinifsayisi + l] == 1:
                        Potential_Yavas_TestSet_Reel.append(l)
            accuracy, accuracy_yavas, recall, precision, fmeasure, cm, weighted_roc_auc_ovo, weighted_roc_auc_ovr,specificity, balanced_accuracy,specificity_for_class_1,specificity_for_class_0 = performance(Potential_Yavas_TestSet_Reel,Test_Yd_hesaplanan_Sinif)
            durum = accuracycheck(accuracy, accuracy_yavas)

        if durum == 0:
            if featureselectioncount <= featureselectionnumber:
                SelecetedFeatures_index = []
                featureselectioncount = featureselectioncount + 1
        else:
            Features_index_const = SelecetedFeatures_index
            durum = 1
        if featureselectioncount == 10:
            break
        print("durum", durum)
        print("featureselectioncount", featureselectioncount)
    Potentialindex_ALL += Potential_Yavas
    Potentialpredict_ALL += Test_Yd_hesaplanan_Sinif
    Accuracy_List.append(accuracy)
    Accuracy_yavas_List.append(accuracy_yavas)
    Precision_List.append(precision)
    Recall_List.append(recall)
    F_Measure_List.append(fmeasure)
    roc_auc_ovo_List.append(weighted_roc_auc_ovo)
    roc_auc_ovr_List.append(weighted_roc_auc_ovr)
    Specificity_List.append(specificity)
    Balanced_accuracy_List.append(balanced_accuracy)
    specificity_for_class_1_List.append(specificity_for_class_1)
    specificity_for_class_0_List.append(specificity_for_class_0)
    nf2 = dt.datetime.now()
# FLOW #

for i in range(1, numberofinput + 1):
    if i in Test_Set[1:, 0]:
        if i in Potentialindex_ALL:
            index = Potentialindex_ALL.index(i)
            predict_ALL.append(Potentialpredict_ALL[index])
        else:
            predict_ALL.append(0)

nS = dt.datetime.now()

print("")
print("PERFORMANS METRİKLERİ")
print("Accuracy: ", Accuracy_List)
print("Accuracy_Yavas: ", Accuracy_yavas_List)
print("Precision_Ort: ", Precision_List)
print("Recall_Ort: ", Recall_List)
print("F-Meause_Ort: ", F_Measure_List)
print("ROC_AUC_OVO: ", roc_auc_ovo_List)
print("ROC_AUC_OVR: ", roc_auc_ovr_List)
print("Specificity: ", Specificity_List)
print("Balanced_accuracy: ", Balanced_accuracy_List)
print("Specificity_for_class_1: ", specificity_for_class_1_List)
print("Specificity_for_class_0: ", specificity_for_class_0_List)
print(" ")
print("PERFORMANS METRİKLERİ - ORT")
print("Accuracy_Ort: ", round(np.mean(Accuracy_List),3))
print("Accuracy_Yavas_Ort: ", round(np.mean(Accuracy_yavas_List),3))
print("Precision_Ort: ", round(np.mean(Precision_List),3))
print("Recall_Ort: ", round(np.mean(Recall_List),3))
print("F-Meause_Ort: ", round(np.mean(F_Measure_List),3))
print("ROC_AUC_OVO_Ort: ", round(np.mean(roc_auc_ovo_List),3))
print("ROC_AUC_OVR_Ort: ", round(np.mean(roc_auc_ovr_List),3))
print("Specificity_Ort: ", round(np.mean(Specificity_List),3))
print("Balanced_accuracy_Ort: ", round(np.mean(Balanced_accuracy_List),3))
print("Specificity_for_class_1_Ort: ", round(np.mean(specificity_for_class_1_List),3))
print("Specificity_for_class_0_Ort: ", round(np.mean(specificity_for_class_0_List),3))

print("Egitim iterasyon sayısı", egitim_itr_sayisi)
print("Sure RBFN Egitim:", n2 - n1)
print("Sure Test", nf2 - nf1)
print("Sure Toplam:", nS - n0)
print(" ")
